#!/usr/bin/env python3

import argparse
import csv
import math
import sys
from pathlib import Path


TARGETS = [
    ("1/8", 1.0 / 8.0),
    ("1/4", 1.0 / 4.0),
    ("1/2", 1.0 / 2.0),
    ("1", 1.0),
    ("2", 2.0),
    ("4", 4.0),
    ("8", 8.0),
]


def find_froc_file(path):
    path = Path(path)
    if path.is_file():
        return path

    candidates = sorted(path.glob("froc_*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No froc_*.txt file found in {path}")
    if len(candidates) > 1:
        result_candidates = [p for p in candidates if p.name.startswith("froc_result_")]
        if len(result_candidates) == 1:
            return result_candidates[0]
        raise RuntimeError(f"Multiple froc_*.txt files found in {path}: {', '.join(p.name for p in candidates)}")
    return candidates[0]


def read_froc_curve(path):
    points = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                fp_rate = float(row[0])
                sensitivity = float(row[1])
            except ValueError:
                continue
            points.append((fp_rate, sensitivity))

    if not points:
        raise RuntimeError(f"No FROC points found in {path}")

    points.sort(key=lambda x: x[0])
    return points


def interp(points, target):
    if target <= points[0][0]:
        return points[0][1]
    if target >= points[-1][0]:
        return points[-1][1]

    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]

        if math.isclose(target, x0):
            return y0
        if math.isclose(target, x1):
            return y1
        if x0 <= target <= x1:
            if math.isclose(x0, x1):
                return y1
            ratio = (target - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)

    return points[-1][1]


def extract_metrics(path):
    froc_file = find_froc_file(path)
    points = read_froc_curve(froc_file)
    values = [(label, target, interp(points, target)) for label, target in TARGETS]
    cpm = sum(value for _, _, value in values) / len(values)
    return froc_file, values, cpm


def main():
    parser = argparse.ArgumentParser(
        description="Extract LUNA16 FROC sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, and 8 FP/scan, then compute CPM."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="FROC txt file(s), or evaluation score directory/directories containing froc_*.txt.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print CSV format instead of a markdown table.",
    )
    args = parser.parse_args()

    rows = []
    for input_path in args.inputs:
        froc_file, values, cpm = extract_metrics(input_path)
        row = {"name": froc_file.parent.name, "file": str(froc_file), "CPM": cpm}
        for label, _, value in values:
            row[label] = value
        rows.append(row)

    headers = ["name", "1/8", "1/4", "1/2", "1", "2", "4", "8", "CPM"]
    if args.csv:
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=headers,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] if key == "name" else f"{row[key]:.3f}" for key in headers})
        return

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join([row["name"]] + [f"{row[key]:.3f}" for key in headers[1:]])
            + " |"
        )


if __name__ == "__main__":
    main()
