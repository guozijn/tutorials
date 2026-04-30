"""
Lung nodule detection using the lung_nodule_ct_detection MONAI bundle.

Discovers all NII.gz files in an input directory, runs the pre-trained
RetinaNet detector on each file, and writes results in LUNA16 CSV format:

    seriesuid, coordX, coordY, coordZ, diameter_mm

Results are written to the CSV after each scan so progress is preserved
if the process is interrupted.

Example (CLI):
    python3 detect_chest_ct.py --input_dir ~/chest_ct_nii

Via MONAI bundle CLI:
    python3 -m monai.bundle run \\
        --config_file configs/chest_ct_inference.json \\
        --input_dir ~/chest_ct_nii \\
        --output_csv chest_ct_nii.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import os
import sys
import time
from pathlib import Path

import torch

BUNDLE_DIR = Path(os.path.expanduser(
    "~/Code/pulmodex/checkpoints/lung_nodule_ct_detection"
))
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = str(SCRIPT_DIR / "chest_ct_nii.csv")
CSV_HEADER = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "score"]


def _seriesuid_from_path(path: str) -> str:
    seriesuid = os.path.basename(path)
    if seriesuid.endswith(".nii.gz"):
        seriesuid = seriesuid[: -len(".nii.gz")]
    return seriesuid


def _progress_path(output_csv: str) -> str:
    return f"{output_csv}.progress.tsv"


def _flush_file(file_obj) -> None:
    file_obj.flush()
    os.fsync(file_obj.fileno())


def _count_existing_rows(output_csv: str) -> int:
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return 0

    with open(output_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != CSV_HEADER:
            raise ValueError(
                f"{output_csv} has an unexpected header: {header!r}. "
                "Use --overwrite to recreate it."
            )
        return sum(1 for _ in reader)


def _bootstrap_completed_seriesuids(output_csv: str) -> set[str]:
    completed: set[str] = set()
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return completed

    with open(output_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != CSV_HEADER:
            raise ValueError(
                f"{output_csv} has an unexpected header: {header!r}. "
                "Use --overwrite to recreate it."
            )
        for row in reader:
            if row:
                completed.add(row[0])
    return completed


def _load_completed_seriesuids(output_csv: str, overwrite: bool) -> tuple[set[str], bool]:
    progress_path = _progress_path(output_csv)
    if overwrite:
        return set(), False

    if os.path.exists(progress_path) and os.path.getsize(progress_path) > 0:
        completed: set[str] = set()
        with open(progress_path, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if header != ["seriesuid", "num_detections"]:
                raise ValueError(
                    f"{progress_path} has an unexpected header: {header!r}. "
                    "Delete it or use --overwrite."
                )
            for row in reader:
                if row:
                    completed.add(row[0])
        return completed, False

    completed = _bootstrap_completed_seriesuids(output_csv)
    if completed:
        print(
            "No progress sidecar found; bootstrapping resume state from the existing CSV. "
            "This can only infer scans that already have at least one detection.",
            flush=True,
        )
    return completed, bool(completed)


def _open_output_files(output_csv: str, overwrite: bool):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    csv_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    csv_mode = "a" if csv_exists and not overwrite else "w"
    csv_file = open(output_csv, csv_mode, newline="")
    writer = csv.writer(csv_file)
    if csv_mode == "w":
        writer.writerow(CSV_HEADER)
        _flush_file(csv_file)

    progress_path = _progress_path(output_csv)
    progress_exists = os.path.exists(progress_path) and os.path.getsize(progress_path) > 0
    progress_mode = "a" if progress_exists and not overwrite else "w"
    progress_file = open(progress_path, progress_mode, newline="")
    progress_writer = csv.writer(progress_file, delimiter="\t")
    if progress_mode == "w":
        progress_writer.writerow(["seriesuid", "num_detections"])
        _flush_file(progress_file)

    return csv_file, writer, progress_file, progress_writer


def _infer_worker(
    file_paths: list[str],
    bundle_dir: str,
    output_csv: str,
    score_thresh: float,
    nms_thresh: float,
    num_workers: int,
    amp: bool,
    overwrite: bool,
) -> int:
    bundle_dir = Path(bundle_dir)
    sys.path.insert(0, str(bundle_dir))

    from monai.bundle import ConfigParser
    from scripts.detection_inferer import RetinaNetInferer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device}  —  {len(file_paths)} file(s)", flush=True)
    if num_workers:
        print(
            f"Ignoring num_workers={num_workers}; scans are processed serially to reduce peak memory.",
            flush=True,
        )

    parser = ConfigParser()
    parser.read_config(str(bundle_dir / "configs" / "inference.json"))

    # whether_raw_luna16=True selects the ITK-reader preprocessing branch:
    #   - LoadImaged with reader="itkreader" and affine_lps_to_ras=True
    #   - Spacingd resampling to 0.703125 x 0.703125 x 1.25 mm
    # This is consistent with how the model was trained on LUNA16 MHD images.
    # SimpleITK reads NII.gz in LPS convention; affine_lps_to_ras corrects it
    # to RAS, matching the convention used during training.
    parser["bundle_root"] = str(bundle_dir)
    parser["whether_raw_luna16"] = True
    parser["device"] = "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
    parser["test_datalist"] = [{"image": f} for f in file_paths]
    parser["load_pretrain"] = False
    parser["amp"] = amp

    preprocessing = parser.get_parsed_content("preprocessing")
    postprocessing = parser.get_parsed_content("postprocessing")

    network = parser.get_parsed_content("network")
    detector = parser.get_parsed_content("detector")

    # Evaluate detector_ops to apply target-key, score/nms, and sliding-window
    # settings as side effects onto the detector.
    parser.get_parsed_content("detector_ops")

    detector.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=1000,
        nms_thresh=nms_thresh,
        detections_per_img=300,
    )

    # Load checkpoint weights directly (avoids needing an Ignite engine).
    # The bundle's model.pt is a plain state dict (OrderedDict), not wrapped.
    ckpt_path = bundle_dir / "models" / "model.pt"
    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    network.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {ckpt_path}", flush=True)

    network.eval()
    inferer = RetinaNetInferer(detector=detector, force_sliding_window=False)

    completed_seriesuids, bootstrap_progress = _load_completed_seriesuids(
        output_csv, overwrite=overwrite
    )
    pending_paths = [
        path for path in file_paths
        if _seriesuid_from_path(path) not in completed_seriesuids
    ]
    existing_detection_count = 0 if overwrite else _count_existing_rows(output_csv)
    skipped = len(file_paths) - len(pending_paths)
    print(
        f"Resume state: {skipped} completed, {len(pending_paths)} remaining, "
        f"{existing_detection_count} detection row(s) already in CSV.",
        flush=True,
    )

    csv_file, writer, progress_file, progress_writer = _open_output_files(
        output_csv, overwrite=overwrite
    )
    if bootstrap_progress:
        for seriesuid in sorted(completed_seriesuids):
            progress_writer.writerow([seriesuid, ""])
        _flush_file(progress_file)

    detection_count = existing_detection_count
    t0 = time.time()
    total = len(pending_paths)

    with torch.inference_mode():
        for done, file_path in enumerate(pending_paths, start=1):
            filename = os.path.basename(file_path)
            elapsed = time.time() - t0
            eta = 0.0 if done == 0 else elapsed / done * (total - done)
            print(
                f"{done}/{total}  {filename}"
                f"  elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                flush=True,
            )

            batch_item = preprocessing({"image": file_path})
            inputs = [batch_item["image"].to(device)]

            if amp and device.type == "cuda":
                with torch.autocast("cuda"):
                    outputs = inferer(inputs, network)
            else:
                outputs = inferer(inputs, network)
            pred = outputs[0]

            # Merge predictions into the batch dict so postprocessing
            # transforms (ClipBoxToImaged, AffineBoxToWorldCoordinated)
            # can access the image tensor and its metadata.
            post_input = {
                **batch_item,
                "box": pred[detector.target_box_key].to(torch.float32),
                "label": pred[detector.target_label_key],
                "label_scores": pred[detector.pred_score_key].to(torch.float32),
            }
            post_output = postprocessing(post_input)

            seriesuid = _seriesuid_from_path(file_path)

            # Post-processing converts boxes to world coordinates in
            # "cccwhd" format: [center_x, center_y, center_z, w, h, d].
            boxes = post_output["box"].cpu().numpy()
            scores = post_output["label_scores"].cpu().numpy()

            scan_rows = []
            for box, score in zip(boxes, scores):
                cx, cy, cz, w, h, d = box.tolist()
                diameter_mm = (w + h + d) / 3.0
                scan_rows.append([seriesuid, cx, cy, cz, diameter_mm, float(score)])

            writer.writerows(scan_rows)
            _flush_file(csv_file)
            progress_writer.writerow([seriesuid, len(scan_rows)])
            _flush_file(progress_file)
            detection_count += len(scan_rows)
            print(f"  -> {seriesuid}: {len(scan_rows)} detection(s)", flush=True)

            del scan_rows, boxes, scores, post_output, post_input, pred, outputs, inputs, batch_item
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_file.close()
    progress_file.close()
    elapsed = time.time() - t0
    print(f"Finished in {elapsed:.1f}s — {detection_count} detection(s)", flush=True)
    return detection_count


# ---------------------------------------------------------------------------
# Public API — callable from a MONAI bundle config via ConfigParser
# ---------------------------------------------------------------------------

def run_detection(
    input_dir: str = str(Path("~/chest_ct_nii").expanduser()),
    output_csv: str = DEFAULT_OUTPUT,
    bundle_dir: str = str(BUNDLE_DIR),
    score_thresh: float = 0.1,
    nms_thresh: float = 0.22,
    num_workers: int = 1,
    overwrite: bool = False,
) -> int:
    """
    Detect lung nodules in all NII.gz files inside *input_dir* and write
    results to *output_csv* in LUNA16 annotation format.

    Returns the total number of detection rows present in the output CSV after
    this run completes.

    Parameters
    ----------
    input_dir   : directory containing .nii.gz files
    output_csv  : destination CSV file (written per scan as inference progresses)
    bundle_dir  : root directory of the lung_nodule_ct_detection bundle
    score_thresh: minimum confidence score to keep a detection
    nms_thresh  : NMS IoU threshold
    num_workers : retained for CLI compatibility; serial processing ignores it
                  to reduce peak memory usage
    overwrite   : recreate the CSV/progress files instead of resuming
    """
    input_dir = os.path.expanduser(input_dir)
    bundle_dir = str(Path(bundle_dir).expanduser())
    output_csv = os.path.expanduser(output_csv)

    nii_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found in {input_dir}")
    print(f"Found {len(nii_files)} NII.gz file(s) in {input_dir}")

    amp = torch.cuda.is_available()
    print(f"AMP: {amp}")

    total_rows = _infer_worker(
        nii_files, bundle_dir, output_csv,
        score_thresh, nms_thresh, num_workers, amp, overwrite,
    )
    print(f"\nSaved {total_rows} detection(s) to {output_csv}")
    return total_rows


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Lung nodule detection on a directory of NII.gz chest CT files.\n\n"
            "Uses the lung_nodule_ct_detection MONAI bundle (RetinaNet trained on\n"
            "LUNA16) to detect pulmonary nodules. Each input file is resampled to\n"
            "0.703 x 0.703 x 1.25 mm before inference. Results are written in\n"
            "LUNA16 annotation CSV format:\n"
            "  seriesuid, coordX, coordY, coordZ, diameter_mm\n\n"
            "The CSV is written after each scan so progress is preserved if the\n"
            "process is interrupted.\n\n"
            "Examples:\n"
            "  # Run on all files in a directory\n"
            "  python3 detect_chest_ct.py --input_dir ~/chest_ct_nii\n\n"
            "  # Custom output path and lower score threshold for higher recall\n"
            "  python3 detect_chest_ct.py \\\n"
            "      --input_dir ~/chest_ct_nii \\\n"
            "      --output_csv ~/results/annotations.csv \\\n"
            "      --score_thresh 0.05\n\n"
            "  # Via MONAI bundle CLI\n"
            "  python3 -m monai.bundle run \\\n"
            "      --config_file configs/chest_ct_inference.json \\\n"
            "      --input_dir ~/chest_ct_nii \\\n"
            "      --output_csv chest_ct_nii.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input_dir",
        default=str(Path("~/chest_ct_nii").expanduser()),
        help=(
            "Directory containing .nii.gz chest CT files. "
            "All files matching *.nii.gz are processed. "
            "(default: ~/chest_ct_nii)"
        ),
    )
    p.add_argument(
        "--output_csv",
        default=DEFAULT_OUTPUT,
        help=(
            "Destination CSV file with columns: "
            "seriesuid, coordX, coordY, coordZ, diameter_mm, score. "
            "Parent directories are created automatically. "
            "Written incrementally after each scan. "
            "(default: <script_dir>/chest_ct_nii.csv)"
        ),
    )
    p.add_argument(
        "--bundle_dir",
        default=str(BUNDLE_DIR),
        help=(
            "Root directory of the lung_nodule_ct_detection MONAI bundle "
            "containing configs/, models/, and scripts/ subdirectories. "
            f"(default: {BUNDLE_DIR})"
        ),
    )
    p.add_argument(
        "--score_thresh",
        type=float,
        default=0.1,
        help=(
            "Minimum confidence score to keep a detection (0.0-1.0). "
            "Lower values increase recall but add more false positives. "
            "Recommended ranges: 0.02-0.05 for maximum recall (screening), "
            "0.1 for balanced annotation, 0.3+ for high-precision output. "
            "The LUNA16 bundle default is 0.02. "
            "(default: 0.1)"
        ),
    )
    p.add_argument(
        "--nms_thresh",
        type=float,
        default=0.22,
        help=(
            "Non-maximum suppression IoU threshold (0.0-1.0). "
            "Lower values suppress more overlapping boxes; "
            "higher values allow more overlapping detections to coexist. "
            "(default: 0.22)"
        ),
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Deprecated compatibility flag. The script now processes one scan "
            "at a time and ignores this value to minimize peak memory usage. "
            "(default: 1)"
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Start a fresh run by recreating the CSV and progress sidecar "
            "instead of resuming from existing files."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_detection(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        bundle_dir=args.bundle_dir,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
