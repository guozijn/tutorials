"""
Per-scan coordinate diagnostic.

Your CTs appear to have heterogeneous orientations (Z ranges from -1100 to
+2045 across scans). This script checks, FOR EACH SCAN, whether the GT box
centers land inside the image and on plausible tissue, under the real
val pipeline, for BOTH affine_lps_to_ras = True and False.

It tells you, per scan, which convention is correct -- or whether the
coordinates simply don't match the images at all.

Run this; paste the summary table back.
"""

import json
import numpy as np
import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd,
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated, ConvertBoxToStandardModed, StandardizeEmptyBoxd,
)

# -------- EDIT THESE --------
DATA_BASE_DIR = "<PATH_TO>/lung_project/data"
DATASET_JSON  = "<PATH_TO>/lung_project/data/dataset.json"
GT_BOX_MODE   = "cccwhd"
# ----------------------------


def pipeline(lps_to_ras):
    return Compose([
        LoadImaged(keys=["image"], image_only=False, meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image", "box"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
        StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ConvertBoxToStandardModed(box_keys=["box"], mode=GT_BOX_MODE),
        AffineBoxToImageCoordinated(
            box_keys=["box"], box_ref_image_keys="image",
            image_meta_key_postfix="meta_dict", affine_lps_to_ras=lps_to_ras,
        ),
    ])


def score_boxes(img, boxes_xyzxyz):
    """Return (in_bounds_count, tissue_count, total) for these image-coord boxes."""
    H, W, D = img.shape
    n = len(boxes_xyzxyz)
    inb = 0
    tis = 0
    for b in boxes_xyzxyz:
        cx = (b[0] + b[3]) / 2
        cy = (b[1] + b[4]) / 2
        cz = (b[2] + b[5]) / 2
        ix, iy, iz = int(round(cx)), int(round(cy)), int(round(cz))
        if 0 <= ix < H and 0 <= iy < W and 0 <= iz < D:
            inb += 1
            r = 3
            patch = img[max(0, ix-r):ix+r, max(0, iy-r):iy+r, max(0, iz-r):iz+r]
            if patch.size:
                m = float(patch.mean())
                # oriented image is raw HU here (intensity transform not applied)
                if -700 < m < 300:
                    tis += 1
    return inb, tis, n


def main():
    with open(DATASET_JSON) as f:
        ds = json.load(f)

    items = []
    for split in ds:
        for it in ds[split]:
            items.append((split, it))

    print(f"{'scan':<40} {'conv':>5} {'inB':>5} {'tissue':>7} {'tot':>4}")
    print("-" * 70)

    verdict = {"True": 0, "False": 0, "neither": 0}

    for split, it in items:
        sid = it["image"].split("/")[-1].replace(".nii.gz", "")[:38]
        best = {}
        for lps in [True, False]:
            data = {
                "image": f"{DATA_BASE_DIR}/{it['image']}",
                "box": np.array(it["box"], dtype=np.float32),
                "label": np.array(it["label"], dtype=np.int64),
            }
            try:
                out = pipeline(lps)(data)
                img = out["image"][0].numpy()
                boxes = np.asarray(out["box"])
                inb, tis, tot = score_boxes(img, boxes)
            except Exception as e:
                inb, tis, tot = -1, -1, len(it["box"])
            best[str(lps)] = (inb, tis, tot)
            print(f"{sid:<40} {str(lps):>5} {inb:>5} {tis:>7} {tot:>4}")

        # decide which convention is better for this scan (more tissue hits)
        t_true = best["True"][1]
        t_false = best["False"][1]
        if t_true == t_false:
            verdict["neither" if t_true == 0 else ("True" if t_true>0 else "neither")] += 1
        elif t_true > t_false:
            verdict["True"] += 1
        else:
            verdict["False"] += 1
        print()

    print("=" * 70)
    print("PER-SCAN VERDICT (which convention puts GT centers on tissue):")
    print(f"  True  better: {verdict['True']} scans")
    print(f"  False better: {verdict['False']} scans")
    print(f"  neither works: {verdict['neither']} scans")
    print()
    if verdict["True"] > 0 and verdict["False"] > 0:
        print(">>> MIXED: scans need different conventions. This means the")
        print(">>> annotation coords and images are NOT in one consistent system.")
        print(">>> The fix is to regenerate coords per-scan from the bundle's")
        print(">>> own world->image handling, not a single global flag.")
    elif verdict["False"] >= verdict["True"]:
        print(">>> Use affine_lps_to_ras=False everywhere.")
    else:
        print(">>> Use affine_lps_to_ras=True everywhere.")


if __name__ == "__main__":
    main()
