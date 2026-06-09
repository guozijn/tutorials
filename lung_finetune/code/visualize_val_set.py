"""
Batch validation visualizer.

Loads the best fine-tuned model, runs inference on every validation scan,
and saves a multi-panel PNG per scan showing GT boxes (red) and predicted
boxes (green/yellow by score) on the axial slice at each box center.

This lets you confirm, scan by scan, which predictions are true positives
(box on a small nodule inside aerated lung) vs false positives (box on
solid soft tissue / bone / vessel).

Output: one PNG per validation scan in OUTPUT_DIR/val_vis/.
Run after at least one checkpoint has been saved (model_path or *_last.pt).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from monai.transforms import ScaleIntensityRanged
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet, resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets import resnet
from monai.utils import set_determinism

from generate_transforms import generate_detection_val_transform


# -------- EDIT THESE --------
ENV_FILE    = "./environment.json"
CONFIG_FILE = "./config_train.json"
USE_LAST    = False   # True -> use *_last.pt, False -> use best model_path
SCORE_DISPLAY_THRESH = 0.1   # draw any prediction above this score
# ----------------------------


def cccwhd_to_xyzxyz(b):
    cx, cy, cz, w, h, d = b
    return [cx - w/2, cy - h/2, cz - d/2, cx + w/2, cy + h/2, cz + d/2]


def draw_boxes_on_slice(ax, image_3d, boxes_xyzxyz, color, scores=None):
    """Draw each box on the axial slice at its own center-z."""
    for i, b in enumerate(boxes_xyzxyz):
        cz = int(round((b[2] + b[5]) / 2))
        cz = max(0, min(image_3d.shape[2] - 1, cz))
        w = b[3] - b[0]
        h = b[4] - b[1]
        rect = patches.Rectangle(
            (b[0], b[1]), w, h, fill=False, edgecolor=color, linewidth=1.5
        )
        ax.add_patch(rect)
        label = ""
        if scores is not None:
            label = f"{scores[i]:.2f}"
        if label:
            ax.text(b[0], b[1] - 3, label, color=color, fontsize=7)


def main():
    set_determinism(seed=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)

    with open(ENV_FILE) as f:
        env = json.load(f)
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)

    out_dir = os.path.join(os.path.dirname(env["model_path"]), "val_vis")
    os.makedirs(out_dir, exist_ok=True)

    # Transforms
    intensity = ScaleIntensityRanged(
        keys=["image"], a_min=-1024, a_max=300.0, b_min=0.0, b_max=1.0, clip=True
    )
    val_tf = generate_detection_val_transform(
        "image", "box", "label", cfg["gt_box_mode"], intensity,
        affine_lps_to_ras=True, amp=False,
    )

    # Data
    val_data = load_decathlon_datalist(
        env["data_list_file_path"], is_segmentation=True,
        data_list_key="validation", base_dir=env["data_base_dir"],
    )
    val_ds = Dataset(data=val_data, transform=val_tf)
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=2,
        collate_fn=no_collation, pin_memory=False,
    )

    # Rebuild detector skeleton (must match training config)
    anchor_gen = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(cfg["returned_layers"]) + 1)],
        base_anchor_shapes=cfg["base_anchor_shapes"],
    )
    conv1_t_size = [max(7, 2*s + 1) for s in cfg["conv1_t_stride"]]
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck, layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=cfg["n_input_channels"],
        conv1_t_stride=cfg["conv1_t_stride"], conv1_t_size=conv1_t_size,
    )
    fe = resnet_fpn_feature_extractor(
        backbone=backbone, spatial_dims=cfg["spatial_dims"],
        pretrained_backbone=False, trainable_backbone_layers=None,
        returned_layers=cfg["returned_layers"],
    )
    num_anchors = anchor_gen.num_anchors_per_location()[0]
    size_divisible = [s * 2 * 2**max(cfg["returned_layers"]) for s in fe.body.conv1.stride]

    # Load trained weights
    model_file = env["model_path"][:-3] + "_last.pt" if USE_LAST else env["model_path"]
    print(f"Loading model: {model_file}")
    net = torch.jit.load(model_file, map_location=device)

    detector = RetinaNetDetector(network=net, anchor_generator=anchor_gen, debug=False).to(device)
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=SCORE_DISPLAY_THRESH, topk_candidates_per_level=1000,
        nms_thresh=cfg["nms_thresh"], detections_per_img=100,
    )
    detector.set_sliding_window_inferer(
        roi_size=cfg["val_patch_size"], overlap=0.25, sw_batch_size=1,
        mode="constant", device="cpu",
    )
    detector.eval()

    print(f"Visualizing {len(val_ds)} validation scans -> {out_dir}\n")

    with torch.no_grad():
        for scan_idx, val_data_batch in enumerate(val_loader):
            for val_data_i in val_data_batch:
                gt_boxes = np.asarray(val_data_i["box"])  # xyzxyz, image coords
                img = val_data_i["image"][0].cpu().numpy()  # (H, W, D)
                val_input = [val_data_i["image"].to(device)]

                use_inferer = not (val_data_i["image"][0].numel() < np.prod(cfg["val_patch_size"]))
                outs = detector(val_input, use_inferer=use_inferer)[0]

                pred_boxes = outs[detector.target_box_key].cpu().numpy()
                pred_scores = outs[detector.pred_score_key].cpu().numpy()

                # One panel per GT box (so each nodule gets its center slice),
                # plus a final panel for predictions with no nearby GT.
                n_gt = len(gt_boxes)
                n_panels = max(1, n_gt)
                ncols = min(3, n_panels)
                nrows = (n_panels + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
                if n_panels == 1:
                    axes = np.array([axes])
                axes = axes.flatten()

                gt_xyz = [list(b) for b in gt_boxes]  # already xyzxyz
                pred_xyz = [list(b) for b in pred_boxes]

                for p in range(n_panels):
                    ax = axes[p]
                    if n_gt > 0:
                        gb = gt_boxes[p]
                        cz = int(round((gb[2] + gb[5]) / 2))
                    else:
                        cz = img.shape[2] // 2
                    cz = max(0, min(img.shape[2]-1, cz))
                    ax.imshow(img[:, :, cz].T, cmap="gray", origin="lower",
                              vmin=0, vmax=1)
                    # GT in red (only those whose center is on/near this slice)
                    for gb in gt_xyz:
                        gcz = (gb[2] + gb[5]) / 2
                        if abs(gcz - cz) <= 5:
                            w, h = gb[3]-gb[0], gb[4]-gb[1]
                            ax.add_patch(patches.Rectangle(
                                (gb[0], gb[1]), w, h, fill=False,
                                edgecolor="red", linewidth=2))
                    # Predictions in lime, label with score
                    for pb, ps in zip(pred_xyz, pred_scores):
                        pcz = (pb[2] + pb[5]) / 2
                        if abs(pcz - cz) <= 5:
                            w, h = pb[3]-pb[0], pb[4]-pb[1]
                            ax.add_patch(patches.Rectangle(
                                (pb[0], pb[1]), w, h, fill=False,
                                edgecolor="lime", linewidth=1.5))
                            ax.text(pb[0], pb[1]-3, f"{ps:.2f}",
                                    color="lime", fontsize=8)
                    ax.set_title(f"slice z={cz}", fontsize=9)
                    ax.axis("off")

                for p in range(n_panels, len(axes)):
                    axes[p].axis("off")

                sid = val_data_i["image"].meta.get("filename_or_obj", f"scan{scan_idx}")
                sid = os.path.basename(str(sid)).replace(".nii.gz", "")
                fig.suptitle(
                    f"{sid}\nGT (red) = {n_gt} | Pred (lime) = {len(pred_xyz)}",
                    fontsize=11,
                )
                outpng = os.path.join(out_dir, f"val_{scan_idx:02d}_{sid[:30]}.png")
                plt.tight_layout()
                plt.savefig(outpng, dpi=90, bbox_inches="tight")
                plt.close()
                print(f"  scan {scan_idx}: {sid[:40]} | GT={n_gt} Pred={len(pred_xyz)} -> {os.path.basename(outpng)}")

    print(f"\nDone. Open the PNGs in: {out_dir}")
    print("Red box on a small white spot inside black lung = true nodule.")
    print("Lime box on solid white tissue / bone with no red = false positive.")


if __name__ == "__main__":
    main()
