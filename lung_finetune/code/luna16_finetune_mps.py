# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

import argparse
import gc
import json
import logging
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from generate_transforms import (
    generate_detection_train_transform,
    generate_detection_val_transform,
)
from visualize_image import visualize_one_xy_slice_in_3d_image
from warmup_scheduler import GradualWarmupScheduler

import monai
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import DataLoader, Dataset, box_utils, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.networks.nets import resnet
from monai.transforms import ScaleIntensityRanged
from monai.utils import set_determinism


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_pretrained_weights(net, pretrained_model_path):
    if not pretrained_model_path:
        print("No pretrained model provided; training from random initialization.")
        return

    pretrained = torch.jit.load(pretrained_model_path, map_location="cpu")
    pretrained_sd = pretrained.state_dict()
    current_sd = net.state_dict()

    # Filter out keys with shape mismatch (e.g. head layers when anchor count changed)
    compatible = {}
    skipped_shape = []
    skipped_missing = []

    for k, v in pretrained_sd.items():
        if k not in current_sd:
            skipped_missing.append(k)
        elif v.shape != current_sd[k].shape:
            skipped_shape.append(f"  {k}: pretrained={tuple(v.shape)} vs current={tuple(current_sd[k].shape)}")
        else:
            compatible[k] = v

    net.load_state_dict(compatible, strict=False)

    print(f"Loaded pretrained weights from: {pretrained_model_path}")
    print(f"  Loaded:  {len(compatible)} / {len(pretrained_sd)} tensors")
    if skipped_shape:
        print(f"  Skipped (shape mismatch, will use random init) - {len(skipped_shape)} tensors:")
        for s in skipped_shape:
            print(s)
    if skipped_missing:
        print(f"  Skipped (not in current model): {len(skipped_missing)} tensors")

    del pretrained


def set_all_params_trainable(net):
    total = sum(p.numel() for p in net.parameters())
    for p in net.parameters():
        p.requires_grad = True
    print(f"All parameters set to trainable. Total: {total:,}")


def build_optimizer(net, lr):
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)


# ---------------------------------------------------------------------------
# FROC computation
# ---------------------------------------------------------------------------

def compute_froc(
    pred_boxes_list,
    pred_scores_list,
    gt_boxes_list,
    fps_per_scan_thresholds,
    iou_threshold=0.1,
    num_images=None,
):
    """
    Compute FROC sensitivity at fixed FP-per-scan levels.

    Args:
        pred_boxes_list: list of np.ndarray, shape (N, 6), world or image coords
        pred_scores_list: list of np.ndarray, shape (N,)
        gt_boxes_list: list of np.ndarray, shape (M, 6)
        fps_per_scan_thresholds: list of float, e.g. [0.125, 0.25, 0.5, 1, 2, 4, 8]
        iou_threshold: IoU threshold to consider a detection a true positive
        num_images: total number of scans (defaults to len(gt_boxes_list))

    Returns:
        dict with key "FROC_mean_sens" and per-FP-level sensitivities
    """
    if num_images is None:
        num_images = len(gt_boxes_list)

    total_gt = sum(len(g) for g in gt_boxes_list)
    if total_gt == 0:
        return {"FROC_mean_sens": 0.0}

    # Flatten all predictions with image index
    all_scores = []
    all_tp_flags = []

    for img_idx, (pboxes, pscores, gboxes) in enumerate(
        zip(pred_boxes_list, pred_scores_list, gt_boxes_list)
    ):
        if len(pboxes) == 0:
            continue

        gt_matched = np.zeros(len(gboxes), dtype=bool)

        # Sort by score descending for greedy matching
        order = np.argsort(-pscores)
        pboxes_sorted = pboxes[order]
        pscores_sorted = pscores[order]

        tp_flags = np.zeros(len(pscores_sorted), dtype=bool)

        for det_idx, (pbox, pscore) in enumerate(zip(pboxes_sorted, pscores_sorted)):
            if len(gboxes) == 0:
                all_scores.append(pscore)
                all_tp_flags.append(False)
                continue

            # Compute IoU with all GTs
            p = pbox[None, :]  # (1, 6)
            g = gboxes          # (M, 6)

            # xyzxyz IoU for cccwhd boxes: convert to xyzxyz first
            def cccwhd_to_xyzxyz(b):
                cx, cy, cz, w, h, d = b[..., 0], b[..., 1], b[..., 2], b[..., 3], b[..., 4], b[..., 5]
                return np.stack([cx-w/2, cy-h/2, cz-d/2, cx+w/2, cy+h/2, cz+d/2], axis=-1)

            p_xyz = cccwhd_to_xyzxyz(p)
            g_xyz = cccwhd_to_xyzxyz(g)

            inter_min = np.maximum(p_xyz[:, None, :3], g_xyz[None, :, :3])
            inter_max = np.minimum(p_xyz[:, None, 3:], g_xyz[None, :, 3:])
            inter_dims = np.maximum(0, inter_max - inter_min)
            inter_vol = inter_dims[..., 0] * inter_dims[..., 1] * inter_dims[..., 2]

            p_vol = (p_xyz[:, 3]-p_xyz[:, 0]) * (p_xyz[:, 4]-p_xyz[:, 1]) * (p_xyz[:, 5]-p_xyz[:, 2])
            g_vol = (g_xyz[:, 3]-g_xyz[:, 0]) * (g_xyz[:, 4]-g_xyz[:, 1]) * (g_xyz[:, 5]-g_xyz[:, 2])

            iou = inter_vol / (p_vol[:, None] + g_vol[None, :] - inter_vol + 1e-6)
            iou = iou[0]  # shape (M,)

            best_gt = np.argmax(iou)
            if iou[best_gt] >= iou_threshold and not gt_matched[best_gt]:
                gt_matched[best_gt] = True
                tp_flags[det_idx] = True

            all_scores.append(pscore)
            all_tp_flags.append(tp_flags[det_idx])

    if len(all_scores) == 0:
        return {"FROC_mean_sens": 0.0}

    all_scores = np.array(all_scores)
    all_tp_flags = np.array(all_tp_flags)

    # Sort by score descending
    order = np.argsort(-all_scores)
    all_scores = all_scores[order]
    all_tp_flags = all_tp_flags[order]

    cum_tp = np.cumsum(all_tp_flags)
    cum_fp = np.cumsum(~all_tp_flags)

    sensitivity_at_fps = {}
    for fps_thresh in fps_per_scan_thresholds:
        max_fp = fps_thresh * num_images
        # Find last index where cum_fp <= max_fp
        valid = np.where(cum_fp <= max_fp)[0]
        if len(valid) == 0:
            sens = 0.0
        else:
            sens = float(cum_tp[valid[-1]]) / total_gt
        sensitivity_at_fps[f"FROC_sens_FP{fps_thresh}"] = sens

    mean_sens = float(np.mean(list(sensitivity_at_fps.values())))
    result = {"FROC_mean_sens": mean_sens}
    result.update(sensitivity_at_fps)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RetinaNet fine-tuning for lung nodule detection on Apple Silicon"
    )
    parser.add_argument("-e", "--environment-file", default="./config/environment.json")
    parser.add_argument("-c", "--config-file", default="./config/config_train.json")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("-p", "--pretrained-model", default=None)
    args = parser.parse_args()

    set_determinism(seed=0)

    # Device: MPS for Apple Silicon, fall back to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    amp = False  # AMP not stable on MPS
    print(f"Device: {device} | AMP: {amp}")
    torch.set_num_threads(4)

    with open(args.environment_file, "r") as f:
        env_dict = json.load(f)
    with open(args.config_file, "r") as f:
        config_dict = json.load(f)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    # ------------------------------------------------------------------
    # 1. Transforms
    # ------------------------------------------------------------------
    intensity_transform = ScaleIntensityRanged(
        keys=["image"], a_min=-1024, a_max=300.0, b_min=0.0, b_max=1.0, clip=True
    )

    train_transforms = generate_detection_train_transform(
        "image", "box", "label",
        args.gt_box_mode,
        intensity_transform,
        args.patch_size,
        args.batch_size,
        point_key="points",
        affine_lps_to_ras=True,
        amp=amp,
    )

    val_transforms = generate_detection_val_transform(
        "image", "box", "label",
        args.gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,
        amp=amp,
    )

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    train_data = load_decathlon_datalist(
        args.data_list_file_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=args.data_base_dir,
    )
    try:
        val_data = load_decathlon_datalist(
            args.data_list_file_path,
            is_segmentation=True,
            data_list_key="validation",
            base_dir=args.data_base_dir,
        )
    except (KeyError, ValueError):
        val_data = []

    if not val_data:
        split_idx = int(0.8 * len(train_data))
        train_data, val_data = train_data[:split_idx], train_data[split_idx:]

    print(f"Dataset split -> train: {len(train_data)} CTs | val: {len(val_data)} CTs")
    print(f"  Train nodules: {sum(len(x['box']) for x in train_data)}")
    print(f"  Val nodules:   {sum(len(x['box']) for x in val_data)}")

    train_ds = Dataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        collate_fn=no_collation,
        persistent_workers=True,
    )

    val_ds = Dataset(data=val_data, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
        collate_fn=no_collation,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(args.returned_layers) + 1)],
        base_anchor_shapes=args.base_anchor_shapes,
    )

    conv1_t_size = [max(7, 2 * s + 1) for s in args.conv1_t_stride]
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=args.n_input_channels,
        conv1_t_stride=args.conv1_t_stride,
        conv1_t_size=conv1_t_size,
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=args.spatial_dims,
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=args.returned_layers,
    )

    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [
        s * 2 * 2 ** max(args.returned_layers)
        for s in feature_extractor.body.conv1.stride
    ]

    net = torch.jit.script(
        RetinaNet(
            spatial_dims=args.spatial_dims,
            num_classes=len(args.fg_labels),
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
    )

    pretrained_path = args.pretrained_model or env_dict.get("pretrained_model_path")
    load_pretrained_weights(net, pretrained_path)

    detector = RetinaNetDetector(
        network=net,
        anchor_generator=anchor_generator,
        debug=args.verbose,
    ).to(device)

    # Full parameter fine-tuning (your classmate's recommendation)
    set_all_params_trainable(detector.network)

    # Sampler: increase pos_fraction slightly to compensate small dataset
    pos_fraction = config_dict.get("balanced_sampler_pos_fraction", 0.15)
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=pos_fraction,
        pool_size=20,
        min_neg=16,
    )
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=args.score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=args.nms_thresh,
        detections_per_img=100,
    )
    detector.set_sliding_window_inferer(
        roi_size=args.val_patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device="cpu",
    )

    # ------------------------------------------------------------------
    # 4. Training components
    # ------------------------------------------------------------------
    fine_tune_lr   = config_dict.get("fine_tune_lr", 5e-5)
    max_epochs     = config_dict.get("fine_tune_max_epochs", 50)
    warmup_epochs  = config_dict.get("warmup_epochs", 5)
    val_interval   = config_dict.get("val_interval", 2)
    grad_accum     = config_dict.get("grad_accum_steps", 4)
    w_cls          = config_dict.get("w_cls", 2.5)
    froc_fps       = config_dict.get("froc_fps_per_scan", [0.125, 0.25, 0.5, 1, 2, 4, 8])

    optimizer = build_optimizer(detector.network, fine_tune_lr)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=warmup_epochs,
        after_scheduler=cosine_scheduler,
    )

    optimizer.zero_grad()
    writer = SummaryWriter(args.tfevent_path)

    # COCO metric for AP (supplementary to FROC)
    coco_metric = COCOMetric(
        classes=["nodule"],
        iou_list=[0.1, 0.3, 0.5],
        max_detection=[100],
    )

    best_froc_mean = 0.0
    best_epoch = -1
    epoch_len = len(train_ds) // train_loader.batch_size

    print(f"\nStarting fine-tuning: {max_epochs} epochs | LR: {fine_tune_lr} | "
          f"warmup: {warmup_epochs} epochs | grad_accum: {grad_accum}")
    print(f"FROC FP/scan levels: {froc_fps}\n")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    for epoch in range(max_epochs):
        print("-" * 60)
        print(f"Epoch {epoch + 1}/{max_epochs}  |  LR: {optimizer.param_groups[0]['lr']:.6f}")
        detector.train()

        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_box_loss = 0.0
        step = 0
        t_start = time.time()

        for batch_data in train_loader:
            step += 1

            inputs = [
                batch_data_ii["image"].to(device)
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]
            targets = [
                dict(
                    label=batch_data_ii["label"].to(device),
                    box=batch_data_ii["box"].to(device),
                )
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]

            outputs = detector(inputs, targets)
            cls_loss = outputs[detector.cls_key]
            box_loss = outputs[detector.box_reg_key]
            loss = (w_cls * cls_loss + box_loss) / grad_accum
            loss.backward()

            if step % grad_accum == 0 or step == epoch_len:
                optimizer.step()
                optimizer.zero_grad()

            raw_total = w_cls * cls_loss.detach().item() + box_loss.detach().item()
            epoch_loss += raw_total
            epoch_cls_loss += cls_loss.detach().item()
            epoch_box_loss += box_loss.detach().item()

            if step % 5 == 0:
                print(
                    f"  Step [{step}/{epoch_len}] "
                    f"total={raw_total:.4f}  "
                    f"cls={cls_loss.item():.4f}  "
                    f"box={box_loss.item():.4f}"
                )

        elapsed = time.time() - t_start
        avg_loss     = epoch_loss     / step
        avg_cls_loss = epoch_cls_loss / step
        avg_box_loss = epoch_box_loss / step

        print(f"Epoch {epoch+1} done in {elapsed:.1f}s  |  "
              f"avg_loss={avg_loss:.4f}  cls={avg_cls_loss:.4f}  box={avg_box_loss:.4f}")

        # TensorBoard - training
        writer.add_scalar("Train/Loss_Total",   avg_loss,     epoch + 1)
        writer.add_scalar("Train/Loss_Cls",     avg_cls_loss, epoch + 1)
        writer.add_scalar("Train/Loss_Box",     avg_box_loss, epoch + 1)
        writer.add_scalar("Train/LR",           optimizer.param_groups[0]["lr"], epoch + 1)

        scheduler.step()

        del inputs, batch_data
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

        # Save latest checkpoint every epoch
        torch.jit.save(detector.network, env_dict["model_path"][:-3] + "_last.pt")

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        if (epoch + 1) % val_interval != 0:
            continue

        detector.eval()
        val_outputs_all = []
        val_targets_all = []
        t_val = time.time()

        with torch.no_grad():
            for val_data in val_loader:
                use_inferer = not all(
                    val_data_i["image"][0].numel() < np.prod(args.val_patch_size)
                    for val_data_i in val_data
                )
                val_inputs = [val_data_i.pop("image").to(device) for val_data_i in val_data]
                val_outs = detector(val_inputs, use_inferer=use_inferer)
                val_outputs_all += val_outs
                val_targets_all += val_data

        # DEBUG: inspect box format/coordinate range for first scan with both pred and gt
        for _i in range(len(val_outputs_all)):
            _pb = val_outputs_all[_i][detector.target_box_key].cpu().numpy()
            _gb = val_targets_all[_i][detector.target_box_key].cpu().numpy()
            if len(_pb) > 0 and len(_gb) > 0:
                print(f"DEBUG scan {_i} pred_boxes[0]: {_pb[0]}")
                print(f"DEBUG scan {_i} gt_boxes[0]  : {_gb[0]}")
                break

        print(f"Validation done in {time.time()-t_val:.1f}s  |  {len(val_outputs_all)} scans")

        # Visualization (first val sample)
        if val_targets_all:
            draw_img = visualize_one_xy_slice_in_3d_image(
                gt_boxes=val_targets_all[0]["box"].cpu().numpy(),
                image=val_inputs[0][0].cpu().numpy(),
                pred_boxes=val_outputs_all[0][detector.target_box_key].cpu().numpy(),
            )
            writer.add_image(
                "Val/XY_slice", draw_img.transpose([2, 1, 0]), epoch + 1
            )

        del val_inputs
        if device.type == "mps":
            torch.mps.empty_cache()

        # -- FROC (primary metric) -------------------------------------
        pred_boxes_list  = [v[detector.target_box_key].cpu().numpy() for v in val_outputs_all]
        pred_scores_list = [v[detector.pred_score_key].cpu().numpy()  for v in val_outputs_all]
        gt_boxes_list    = [v[detector.target_box_key].cpu().numpy()  for v in val_targets_all]

        froc_results = compute_froc(
            pred_boxes_list=pred_boxes_list,
            pred_scores_list=pred_scores_list,
            gt_boxes_list=gt_boxes_list,
            fps_per_scan_thresholds=froc_fps,
            iou_threshold=0.1,
            num_images=len(val_ds),
        )

        froc_mean = froc_results["FROC_mean_sens"]
        print(f"\n  FROC @ Epoch {epoch+1}:")
        for k, v in froc_results.items():
            print(f"    {k}: {v:.4f}")
            writer.add_scalar(f"Val/FROC/{k}", v, epoch + 1)

        # -- COCO AP (supplementary) -----------------------------------
        # Both pred_boxes and gt_boxes are already in xyzxyz format at this point:
        # - pred_boxes: decoded by detector's box_selector (xyzxyz output)
        # - gt_boxes: produced by val_transform pipeline via ConvertBoxToStandardModed -> xyzxyz
        matching_results = matching_batch(
            iou_fn=box_utils.box_iou,
            iou_thresholds=coco_metric.iou_thresholds,
            pred_boxes=pred_boxes_list,
            pred_classes=[v[detector.target_label_key].cpu().numpy() for v in val_outputs_all],
            pred_scores=pred_scores_list,
            gt_boxes=gt_boxes_list,
            gt_classes=[v[detector.target_label_key].cpu().numpy() for v in val_targets_all],
        )
        coco_dict = coco_metric(matching_results)[0]

        ap_primary = coco_dict.get("AP_IoU_0.10_0.50_MaxDet_100", 0.0)
        ar_primary = coco_dict.get("AR_IoU_0.10_0.50_MaxDet_100", 0.0)

        print(f"  COCO AP @[IoU=0.10:0.50]: {ap_primary:.4f}")
        print(f"  COCO AR @[IoU=0.10:0.50]: {ar_primary:.4f}")

        writer.add_scalar("Val/COCO_AP_IoU0.10_0.50", ap_primary, epoch + 1)
        writer.add_scalar("Val/COCO_AR_IoU0.10_0.50", ar_primary, epoch + 1)

        # Count FP per scan at score_thresh
        fp_count = sum(
            (v[detector.pred_score_key].cpu().numpy() >= args.score_thresh).sum()
            - len(gt_boxes_list[i])
            for i, v in enumerate(val_outputs_all)
        )
        # clamp to 0 (can't be negative)
        fp_count = max(0, fp_count)
        fp_per_scan = fp_count / max(1, len(val_ds))
        print(f"  Estimated FP/scan @score_thresh={args.score_thresh}: {fp_per_scan:.2f}")
        writer.add_scalar("Val/FP_per_scan", fp_per_scan, epoch + 1)

        print(f"\n  Best so far: FROC_mean_sens={best_froc_mean:.4f} @ epoch {best_epoch}")

        # Save best model based on FROC
        if froc_mean > best_froc_mean:
            best_froc_mean = froc_mean
            best_epoch = epoch + 1
            torch.jit.save(detector.network, env_dict["model_path"])
            print(f"  >>> New best model saved (FROC_mean_sens={best_froc_mean:.4f}) <<<")

    print("\n" + "=" * 60)
    print(f"Training finished.")
    print(f"Best FROC_mean_sens: {best_froc_mean:.4f} at epoch {best_epoch}")
    writer.close()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
