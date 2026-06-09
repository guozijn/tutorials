"""
Fix coordinate convention in dataset.json by negating X and Y of every box,
then VALIDATE that all annotations land at plausible in-bounds voxel positions.

Background: the annotation/box coordinates were stored in the raw bundle XYZ
convention, but the training pipeline (affine_lps_to_ras=True) expects the
opposite X/Y sign. Manually negating X,Y placed points correctly on nodules
in 3D Slicer. This script applies that negation to ALL boxes and verifies it.

It writes:
  - dataset_fixed.json  (negated X,Y; ready for training)
  - prints a validation report; if everything is in-bounds, you are good.

Box format is cccwhd: [cx, cy, cz, w, h, d]. Only cx, cy are negated
(centers); w,h,d are sizes and stay positive.
"""

import json
import numpy as np

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
)

# -------- EDIT THESE --------
DATA_BASE_DIR = "<PATH_TO>/lung_project/data"
DATASET_JSON  = "<PATH_TO>/lung_project/data/dataset.json"
OUTPUT_JSON   = "<PATH_TO>/lung_project/data/dataset_fixed.json"
VALIDATE      = False  # set True to load each CT and check voxel positions
# ----------------------------


def negate_xy_boxes(boxes):
    """boxes: list of [cx,cy,cz,w,h,d]; negate cx, cy only."""
    out = []
    for b in boxes:
        nb = list(b)
        nb[0] = -nb[0]
        nb[1] = -nb[1]
        out.append(nb)
    return out


def world_to_voxel(world_xyz, affine):
    inv = np.linalg.inv(affine)
    homog = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    return (inv @ homog)[:3]


def main():
    with open(DATASET_JSON) as f:
        ds = json.load(f)

    fixed = {}
    for split in ds:
        fixed[split] = []
        for item in ds[split]:
            new_item = dict(item)
            new_item["box"] = negate_xy_boxes(item["box"])
            fixed[split].append(new_item)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(fixed, f, indent=2)
    print(f"Wrote negated dataset -> {OUTPUT_JSON}\n")

    if not VALIDATE:
        return

    loader = Compose([
        LoadImaged(keys=["image"], image_only=False, meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
    ])

    main.tissue_orig = 0
    main.tissue_fixed = 0
    total = 0
    in_bounds_orig = 0
    in_bounds_fixed = 0
    problems = []

    for split in fixed:
        for item in fixed[split]:
            sid = item["image"].split("/")[-1].replace(".nii.gz", "")
            path = f"{DATA_BASE_DIR}/{item['image']}"
            try:
                data = loader({"image": path})
                vol = data["image"][0].numpy()
                affine = np.array(data["image"].meta["affine"])
            except Exception as e:
                problems.append(f"{sid}: load error {e}")
                continue

            shape = vol.shape

            def local_mean_hu(vox):
                x, y, z = [int(round(v)) for v in vox]
                if not (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
                    return None
                r = 3
                patch = vol[max(0, x-r):x+r, max(0, y-r):y+r, max(0, z-r):z+r]
                return float(patch.mean()) if patch.size else None

            orig_item = next(o for o in ds[split] if o["image"] == item["image"])
            for orig_box, fix_box in zip(orig_item["box"], item["box"]):
                total += 1
                vo = world_to_voxel(orig_box[:3], affine)
                ob = (0 <= vo[0] < shape[0] and 0 <= vo[1] < shape[1] and 0 <= vo[2] < shape[2])
                in_bounds_orig += int(ob)
                hu_o = local_mean_hu(vo)

                vf = world_to_voxel(fix_box[:3], affine)
                fb = (0 <= vf[0] < shape[0] and 0 <= vf[1] < shape[1] and 0 <= vf[2] < shape[2])
                in_bounds_fixed += int(fb)
                hu_f = local_mean_hu(vf)

                # A nodule center should be tissue (HU above air ~ -1000),
                # but NOT dense bone/contrast. Track plausible-tissue hits.
                if hu_o is not None and -700 < hu_o < 300:
                    main.tissue_orig += 1
                if hu_f is not None and -700 < hu_f < 300:
                    main.tissue_fixed += 1

                if not fb:
                    problems.append(
                        f"{sid}: fixed center voxel {vf.round(0)} OUT of {shape}")

    print("=== VALIDATION REPORT ===")
    print(f"Total annotations checked: {total}")
    print(f"In-bounds BEFORE fix (original X,Y): {in_bounds_orig}/{total}")
    print(f"In-bounds AFTER  fix (negated X,Y) : {in_bounds_fixed}/{total}")
    print(f"Center in plausible tissue BEFORE  : {main.tissue_orig}/{total}")
    print(f"Center in plausible tissue AFTER   : {main.tissue_fixed}/{total}")
    print()
    if in_bounds_fixed > in_bounds_orig:
        print("=> Negation INCREASED in-bounds count: fix is consistent.")
    elif in_bounds_fixed == in_bounds_orig == total:
        print("=> Both fully in-bounds. Sign issue is about WHICH SIDE, not bounds.")
        print("   Trust the 3D Slicer check you already did (negation lands on nodule).")
    else:
        print("=> WARNING: negation did not improve bounds. Re-examine before using.")
    if problems:
        print(f"\nIssues ({len(problems)}):")
        for p in problems[:20]:
            print("  " + p)

    print(f"\nIf the report looks good, point environment.json's")
    print(f"data_list_file_path at:\n  {OUTPUT_JSON}")
    print("and retrain. Keep affine_lps_to_ras=True (unchanged).")


if __name__ == "__main__":
    main()
