"""
Single clean detection image for the slide.

Shows ONLY the fine-tuned model on one validation scan, zoomed in on the
nodule, with a full-context axial view plus a zoomed inset. Red = ground
truth, green = prediction with confidence. Designed to be readable on a slide.

Default SCAN_INDEX=0 (a single-nodule scan -> least clutter).
"""

import json
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
ENV_FILE    = "./config/environment.json"
CONFIG_FILE = "./config/config_train.json"
FINETUNED   = "<PATH_TO>/lung_project/model_best_finetuned.pt"
OUTPUT_PNG  = "<PATH_TO>/lung_project/detection_clean.png"
SCAN_INDEX  = 0
SCORE_THRESH = 0.4
ZOOM_HALF   = 50   # voxels around nodule for the inset
# ----------------------------


def build_detector(model_path, cfg, device):
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
    net = torch.jit.load(model_path, map_location=device)
    det = RetinaNetDetector(network=net, anchor_generator=anchor_gen, debug=False).to(device)
    det.set_target_keys(box_key="box", label_key="label")
    det.set_box_selector_parameters(
        score_thresh=SCORE_THRESH, topk_candidates_per_level=1000,
        nms_thresh=cfg["nms_thresh"], detections_per_img=100,
    )
    det.set_sliding_window_inferer(
        roi_size=cfg["val_patch_size"], overlap=0.25, sw_batch_size=1,
        mode="constant", device="cpu",
    )
    det.eval()
    return det


def main():
    set_determinism(seed=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with open(ENV_FILE) as f: env = json.load(f)
    with open(CONFIG_FILE) as f: cfg = json.load(f)

    intensity = ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=300.0,
                                     b_min=0.0, b_max=1.0, clip=True)
    val_tf = generate_detection_val_transform(
        "image", "box", "label", cfg["gt_box_mode"], intensity,
        affine_lps_to_ras=True, amp=False)
    val_data = load_decathlon_datalist(
        env["data_list_file_path"], is_segmentation=True,
        data_list_key="validation", base_dir=env["data_base_dir"])
    ds = Dataset(data=[val_data[SCAN_INDEX]], transform=val_tf)
    ld = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=no_collation, pin_memory=False)

    item = next(iter(ld))[0]
    gt_boxes = np.asarray(item["box"])
    img = item["image"][0].cpu().numpy()
    use_inf = not (item["image"][0].numel() < np.prod(cfg["val_patch_size"]))

    det = build_detector(FINETUNED, cfg, device)
    with torch.no_grad():
        out = det([item["image"].to(device)], use_inferer=use_inf)[0]
    pred_boxes = out[det.target_box_key].cpu().numpy()
    pred_scores = out[det.pred_score_key].cpu().numpy()

    gb = gt_boxes[0]
    cx = int(round((gb[0] + gb[3]) / 2))
    cy = int(round((gb[1] + gb[4]) / 2))
    cz = int(round((gb[2] + gb[5]) / 2))
    cx = max(0, min(img.shape[0]-1, cx))
    cy = max(0, min(img.shape[1]-1, cy))
    cz = max(0, min(img.shape[2]-1, cz))

    # pick the prediction closest to the GT center to display its score
    best_score = None
    if len(pred_boxes) > 0:
        gc = np.array([cx, cy, cz])
        dists = []
        for pb in pred_boxes:
            pc = np.array([(pb[0]+pb[3])/2, (pb[1]+pb[4])/2, (pb[2]+pb[5])/2])
            dists.append(np.linalg.norm(pc - gc))
        best_idx = int(np.argmin(dists))
        best_score = pred_scores[best_idx]

    H = ZOOM_HALF

    def draw_plane(ax, plane2d, gt_rect, pred_rects, title, origin_xy):
        ax.imshow(plane2d, cmap="gray", origin="lower", vmin=0, vmax=1,
                  extent=[origin_xy[0], origin_xy[0]+plane2d.shape[1],
                          origin_xy[1], origin_xy[1]+plane2d.shape[0]])
        gx, gy, gw, gh = gt_rect
        ax.add_patch(patches.Rectangle((gx, gy), gw, gh, fill=False,
                                       edgecolor="red", linewidth=2.5))
        for (px, py, pw, ph) in pred_rects:
            ax.add_patch(patches.Rectangle((px, py), pw, ph, fill=False,
                                           edgecolor="lime", linewidth=2))
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    def crop_plane(plane2d, ca, cb, half):
        A, B = plane2d.shape
        a0, a1 = max(0, ca-half), min(B, ca+half)
        b0, b1 = max(0, cb-half), min(A, cb+half)
        return plane2d[b0:b1, a0:a1], (a0, b0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Axial: img[:,:,cz] -> (X,Y); display as Y rows, X cols => .T
    axial = img[:, :, cz].T
    crop, (ox, oy) = crop_plane(axial, cx, cy, H)
    preds = [(pb[0], pb[1], pb[3]-pb[0], pb[4]-pb[1]) for pb in pred_boxes
             if abs((pb[2]+pb[5])/2 - cz) <= 8]
    draw_plane(axes[0], crop, (gb[0], gb[1], gb[3]-gb[0], gb[4]-gb[1]),
               preds, "Axial", (ox, oy))

    # Coronal: img[:,cy,:] -> (X,Z); display as Z rows, X cols => .T
    coronal = img[:, cy, :].T
    crop, (ox, oy) = crop_plane(coronal, cx, cz, H)
    preds = [(pb[0], pb[2], pb[3]-pb[0], pb[5]-pb[2]) for pb in pred_boxes
             if abs((pb[1]+pb[4])/2 - cy) <= 8]
    draw_plane(axes[1], crop, (gb[0], gb[2], gb[3]-gb[0], gb[5]-gb[2]),
               preds, "Coronal", (ox, oy))

    # Sagittal: img[cx,:,:] -> (Y,Z); display as Z rows, Y cols => .T
    sagittal = img[cx, :, :].T
    crop, (ox, oy) = crop_plane(sagittal, cy, cz, H)
    preds = [(pb[1], pb[2], pb[4]-pb[1], pb[5]-pb[2]) for pb in pred_boxes
             if abs((pb[0]+pb[3])/2 - cx) <= 8]
    draw_plane(axes[2], crop, (gb[1], gb[2], gb[4]-gb[1], gb[5]-gb[2]),
               preds, "Sagittal", (ox, oy))

    score_txt = f"   |   prediction confidence: {best_score:.2f}" if best_score is not None else ""
    fig.suptitle("Fine-tuned model detecting a local nodule (3D views)\n"
                 "Red = radiologist-style ground truth   Green = model prediction" + score_txt,
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved -> {OUTPUT_PNG}")
    if best_score is not None:
        print(f"Nearest prediction confidence: {best_score:.2f}")


if __name__ == "__main__":
    main()
