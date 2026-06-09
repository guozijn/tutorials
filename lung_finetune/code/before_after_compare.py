"""
Before/after fine-tuning comparison on a single validation scan.

Loads the PRETRAINED model and the FINE-TUNED model, runs both on the same
scan, and draws GT (red) + predictions (lime, with score) for each on the
same axial slice, side by side. Use the output PNG in your presentation.

Pick SCAN_INDEX to choose which validation scan to show. Prefer one where
fine-tuning visibly reduces false positives or tightens localization.
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
PRETRAINED  = "<PATH_TO>/lung_project/model_luna16_fold1.pt"
FINETUNED   = "<PATH_TO>/lung_project/model_best_finetuned.pt"
OUTPUT_PNG  = "<PATH_TO>/lung_project/before_after_compare.png"
SCAN_INDEX  = 0       # which validation scan (used when SCAN_ALL=False)
SCAN_ALL    = True    # True: scan all scans and report; False: render SCAN_INDEX
SCORE_THRESH = 0.3    # display threshold; lower shows more predictions/FPs
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


def draw(ax, img, cz, gt_boxes, pred_boxes, pred_scores, title):
    ax.imshow(img[:, :, cz].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    for gb in gt_boxes:
        gcz = (gb[2] + gb[5]) / 2
        if abs(gcz - cz) <= 6:
            ax.add_patch(patches.Rectangle(
                (gb[0], gb[1]), gb[3]-gb[0], gb[4]-gb[1],
                fill=False, edgecolor="red", linewidth=2))
    n_pred_here = 0
    for pb, ps in zip(pred_boxes, pred_scores):
        pcz = (pb[2] + pb[5]) / 2
        if abs(pcz - cz) <= 6:
            n_pred_here += 1
            ax.add_patch(patches.Rectangle(
                (pb[0], pb[1]), pb[3]-pb[0], pb[4]-pb[1],
                fill=False, edgecolor="lime", linewidth=1.5))
            ax.text(pb[0], pb[1]-3, f"{ps:.2f}", color="lime", fontsize=9)
    ax.set_title(f"{title}\nPred on this slice: {n_pred_here}", fontsize=11)
    ax.axis("off")


def scan_all(cfg, env, device):
    """Run both models on every val scan; report prediction-count differences
    so you can pick the most illustrative scan for the slide."""
    intensity = ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=300.0,
                                     b_min=0.0, b_max=1.0, clip=True)
    val_tf = generate_detection_val_transform(
        "image", "box", "label", cfg["gt_box_mode"], intensity,
        affine_lps_to_ras=True, amp=False)
    val_data = load_decathlon_datalist(
        env["data_list_file_path"], is_segmentation=True,
        data_list_key="validation", base_dir=env["data_base_dir"])

    det_pre = build_detector(PRETRAINED, cfg, device)
    det_ft = build_detector(FINETUNED, cfg, device)

    print(f"\n{'idx':>3} {'GT':>4} {'pre_pred':>9} {'ft_pred':>8} {'note'}")
    print("-" * 50)
    for i, d in enumerate(val_data):
        ds = Dataset(data=[d], transform=val_tf)
        ld = DataLoader(ds, batch_size=1, num_workers=2,
                        collate_fn=no_collation, pin_memory=False)
        item = next(iter(ld))[0]
        n_gt = len(np.asarray(item["box"]))
        use_inf = not (item["image"][0].numel() < np.prod(cfg["val_patch_size"]))
        with torch.no_grad():
            n_pre = len(det_pre([item["image"].to(device)], use_inferer=use_inf)[0][det_pre.target_box_key])
            n_ft = len(det_ft([item["image"].to(device)], use_inferer=use_inf)[0][det_ft.target_box_key])
        if device.type == "mps": torch.mps.empty_cache()
        note = ""
        if n_pre > n_ft and n_ft >= n_gt:
            note = "<-- FT reduced FPs (good for slide)"
        elif n_ft > n_pre:
            note = "FT found more"
        print(f"{i:>3} {n_gt:>4} {n_pre:>9} {n_ft:>8} {note}")
    print("\nPick the idx with the biggest pre_pred > ft_pred gap (FT removed false positives),")
    print("set SCAN_INDEX to it, set SCAN_ALL=False, and rerun to render that comparison.\n")


def main():
    set_determinism(seed=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with open(ENV_FILE) as f: env = json.load(f)
    with open(CONFIG_FILE) as f: cfg = json.load(f)

    if SCAN_ALL:
        scan_all(cfg, env, device)
        return

    intensity = ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=300.0,
                                     b_min=0.0, b_max=1.0, clip=True)
    val_tf = generate_detection_val_transform(
        "image", "box", "label", cfg["gt_box_mode"], intensity,
        affine_lps_to_ras=True, amp=False)
    val_data = load_decathlon_datalist(
        env["data_list_file_path"], is_segmentation=True,
        data_list_key="validation", base_dir=env["data_base_dir"])
    val_ds = Dataset(data=[val_data[SCAN_INDEX]], transform=val_tf)
    loader = DataLoader(val_ds, batch_size=1, num_workers=2,
                        collate_fn=no_collation, pin_memory=False)

    batch = next(iter(loader))
    item = batch[0]
    gt_boxes = np.asarray(item["box"])
    img = item["image"][0].cpu().numpy()
    use_inferer = not (item["image"][0].numel() < np.prod(cfg["val_patch_size"]))

    # center slice from first GT
    cz = int(round((gt_boxes[0][2] + gt_boxes[0][5]) / 2)) if len(gt_boxes) else img.shape[2]//2
    cz = max(0, min(img.shape[2]-1, cz))

    results = {}
    for tag, path in [("Pretrained (baseline)", PRETRAINED), ("Fine-tuned (ours)", FINETUNED)]:
        det = build_detector(path, cfg, device)
        with torch.no_grad():
            out = det([item["image"].to(device)], use_inferer=use_inferer)[0]
        results[tag] = (
            out[det.target_box_key].cpu().numpy(),
            out[det.pred_score_key].cpu().numpy(),
        )
        del det
        if device.type == "mps": torch.mps.empty_cache()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (tag, (pb, ps)) in zip(axes, results.items()):
        draw(ax, img, cz, gt_boxes, pb, ps, tag)
    fig.suptitle("Lung nodule detection: before vs after fine-tuning\n"
                 "Red = ground truth, Green = model prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved comparison -> {OUTPUT_PNG}")
    for tag, (pb, ps) in results.items():
        print(f"  {tag}: total predictions = {len(pb)}")


if __name__ == "__main__":
    main()
