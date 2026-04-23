"""
Lung nodule detection using the lung_nodule_ct_detection MONAI bundle.

Discovers all NII.gz files in an input directory, runs the pre-trained
RetinaNet detector on each file, and writes results in LUNA16 CSV format:

    seriesuid, coordX, coordY, coordZ, diameter_mm

Preprocessing is parallelised via DataLoader workers. When multiple GPUs are
available the file list is split evenly across GPUs using separate processes
spawned with torch.multiprocessing.

Example (CLI):
    python3 detect_chest_ct.py --input_dir ~/chest_ct_nii

Multi-GPU:
    python3 detect_chest_ct.py --input_dir ~/chest_ct_nii --num_gpus 4

Via MONAI bundle CLI:
    python3 -m monai.bundle run \\
        --config_file configs/chest_ct_inference.json \\
        --input_dir ~/chest_ct_nii \\
        --output_csv chest_ct_nii.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

BUNDLE_DIR = Path(os.path.expanduser(
    "~/Code/pulmodex/checkpoints/lung_nodule_ct_detection"
))
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = str(SCRIPT_DIR / "chest_ct_nii.csv")


# ---------------------------------------------------------------------------
# Per-GPU worker (runs in a separate process for each GPU)
# ---------------------------------------------------------------------------

def _infer_worker(
    rank: int,
    file_paths: list[str],
    bundle_dir: str,
    score_thresh: float,
    nms_thresh: float,
    num_workers: int,
    amp: bool,
    result_queue,
) -> None:
    """
    Loads the bundle on device *rank*, runs inference on *file_paths*, and
    pushes (rank, rows) into *result_queue*.
    """
    from pathlib import Path

    bundle_dir = Path(bundle_dir)
    # Insert the bundle root so that "scripts" is importable as a package
    # (bundle_root/scripts/__init__.py exists).
    sys.path.insert(0, str(bundle_dir))

    from monai.bundle import ConfigParser
    from monai.data import DataLoader, Dataset
    from monai.data.utils import no_collation
    from scripts.detection_inferer import RetinaNetInferer

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"[GPU {rank}] {device}  —  {len(file_paths)} file(s)", flush=True)

    # ------------------------------------------------------------------
    # Build all bundle components through ConfigParser so the bundle's
    # default transforms and architecture are used without duplication.
    # ------------------------------------------------------------------
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
    parser["device"] = f"$torch.device('cuda:{rank}' if torch.cuda.is_available() else 'cpu')"
    parser["test_datalist"] = [{"image": f} for f in file_paths]
    parser["load_pretrain"] = False   # checkpoint is loaded manually below
    parser["amp"] = amp

    preprocessing = parser.get_parsed_content("preprocessing")
    postprocessing = parser.get_parsed_content("postprocessing")

    dataset = Dataset(data=[{"image": f} for f in file_paths], transform=preprocessing)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=no_collation,
    )

    network = parser.get_parsed_content("network")
    detector = parser.get_parsed_content("detector")

    # Evaluate detector_ops to apply target-key, score/nms, and sliding-window
    # settings as side effects onto the detector.
    parser.get_parsed_content("detector_ops")

    # Allow caller-supplied thresholds to override the bundle defaults.
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
    print(f"[GPU {rank}] Loaded checkpoint: {ckpt_path}", flush=True)

    network.eval()
    inferer = RetinaNetInferer(detector=detector, force_sliding_window=False)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    rows: list[list] = []
    t0 = time.time()
    total = len(file_paths)

    with torch.no_grad():
        for done, batch in enumerate(loader, start=1):
            filenames = [d["image"].meta["filename_or_obj"] for d in batch]
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(
                f"[GPU {rank}] {done}/{total}  {os.path.basename(filenames[0])}"
                f"  elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                flush=True,
            )

            inputs = [d["image"].to(device) for d in batch]

            if amp and device.type == "cuda":
                with torch.autocast("cuda"):
                    outputs = inferer(inputs, network)
            else:
                outputs = inferer(inputs, network)
            del inputs

            for batch_item, pred in zip(batch, outputs):
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

                filename = batch_item["image"].meta["filename_or_obj"]
                seriesuid = os.path.basename(filename)
                if seriesuid.endswith(".nii.gz"):
                    seriesuid = seriesuid[: -len(".nii.gz")]

                # Post-processing converts boxes to world coordinates in
                # "cccwhd" format: [center_x, center_y, center_z, w, h, d].
                boxes = post_output["box"].cpu().numpy()
                scores = post_output["label_scores"].cpu().numpy()

                n = len(boxes)
                print(f"  -> {seriesuid}: {n} detection(s)", flush=True)

                for box, score in zip(boxes, scores):
                    cx, cy, cz, w, h, d = box.tolist()
                    diameter_mm = (w + h + d) / 3.0
                    rows.append([seriesuid, cx, cy, cz, diameter_mm, float(score)])

    elapsed = time.time() - t0
    print(
        f"[GPU {rank}] Finished in {elapsed:.1f}s — {len(rows)} detection(s)",
        flush=True,
    )
    result_queue.put((rank, rows))


# ---------------------------------------------------------------------------
# Public API — callable from a MONAI bundle config via ConfigParser
# ---------------------------------------------------------------------------

def run_detection(
    input_dir: str = str(Path("~/chest_ct_nii").expanduser()),
    output_csv: str = DEFAULT_OUTPUT,
    bundle_dir: str = str(BUNDLE_DIR),
    score_thresh: float = 0.1,
    nms_thresh: float = 0.22,
    num_workers: int = 4,
    num_gpus: int | None = None,
) -> list[list]:
    """
    Detect lung nodules in all NII.gz files inside *input_dir* and write
    results to *output_csv* in LUNA16 annotation format.

    Returns the list of detection rows so callers can do further processing.

    Parameters
    ----------
    input_dir   : directory containing .nii.gz files
    output_csv  : destination CSV file
    bundle_dir  : root directory of the lung_nodule_ct_detection bundle
    score_thresh: minimum confidence score to keep a detection
    nms_thresh  : NMS IoU threshold
    num_workers : DataLoader workers used for parallel preprocessing per GPU
    num_gpus    : number of GPUs to use; None means all available (min 1)
    """
    input_dir = os.path.expanduser(input_dir)
    bundle_dir = str(Path(bundle_dir).expanduser())

    nii_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found in {input_dir}")
    print(f"Found {len(nii_files)} NII.gz file(s) in {input_dir}")

    n_cuda = torch.cuda.device_count()
    n_gpus = num_gpus if num_gpus is not None else max(1, n_cuda)
    n_gpus = min(n_gpus, max(1, n_cuda))
    amp = torch.cuda.is_available()
    print(f"GPUs: {n_gpus}  AMP: {amp}")

    # Distribute files round-robin across GPUs
    chunks: list[list[str]] = [[] for _ in range(n_gpus)]
    for i, f in enumerate(nii_files):
        chunks[i % n_gpus].append(f)

    all_rows: list[list] = []

    if n_gpus == 1:
        # Single GPU: run inline to avoid multiprocessing overhead.
        q: mp.SimpleQueue = mp.SimpleQueue()
        _infer_worker(
            0, chunks[0], bundle_dir,
            score_thresh, nms_thresh, num_workers, amp, q,
        )
        _, rows = q.get()
        all_rows.extend(rows)
    else:
        # Multiple GPUs: spawn one process per GPU (spawn context is required
        # for CUDA safety on Linux where the default is "fork").
        ctx = mp.get_context("spawn")
        q = ctx.SimpleQueue()
        processes = []
        for rank in range(n_gpus):
            if not chunks[rank]:
                continue
            proc = ctx.Process(
                target=_infer_worker,
                args=(
                    rank, chunks[rank], bundle_dir,
                    score_thresh, nms_thresh, num_workers, amp, q,
                ),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

        # Collect results, preserving rank order for deterministic output.
        results: dict[int, list] = {}
        while not q.empty():
            rank, rows = q.get()
            results[rank] = rows
        for rank in sorted(results):
            all_rows.extend(results[rank])

    # Write CSV
    output_csv = os.path.expanduser(output_csv)
    csv_dir = os.path.dirname(os.path.abspath(output_csv))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm", "score"])
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} detection(s) to {output_csv}")
    return all_rows


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
            "Examples:\n"
            "  # Run on all files in a directory (single GPU, default settings)\n"
            "  python3 detect_chest_ct.py --input_dir ~/chest_ct_nii\n\n"
            "  # Custom output path and lower score threshold for higher recall\n"
            "  python3 detect_chest_ct.py \\\n"
            "      --input_dir ~/chest_ct_nii \\\n"
            "      --output_csv ~/results/annotations.csv \\\n"
            "      --score_thresh 0.05\n\n"
            "  # Use all 4 GPUs with 8 preprocessing workers each\n"
            "  python3 detect_chest_ct.py \\\n"
            "      --input_dir ~/chest_ct_nii \\\n"
            "      --num_gpus 4 \\\n"
            "      --num_workers 8\n\n"
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
        default=4,
        help=(
            "Number of DataLoader worker processes for parallel file loading "
            "and preprocessing (resampling, intensity scaling) per GPU. "
            "Increase on machines with many CPU cores to reduce GPU idle time. "
            "(default: 4)"
        ),
    )
    p.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help=(
            "Number of GPUs to use. Files are split evenly across GPUs and "
            "each GPU runs in a separate process (spawn context). "
            "Defaults to all available GPUs, with a minimum of 1. "
            "(default: all available)"
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
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
