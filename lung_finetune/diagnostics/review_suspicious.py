"""
Tri-planar review montage for SUSPICIOUS annotations only.

Reads suspicious_checklist.csv (52 annotations across 22 scans flagged by the
per-scan HU diagnostic) and renders, for each, a zoomed axial+coronal+sagittal
crop centered on the point with a red crosshair.

IMPORTANT: uses affine_lps_to_ras=True implicitly (it loads the image in RAS
via Orientationd and maps the LPS world coord with sign flip), matching the
convention confirmed correct by the per-scan diagnostic.

Each row is labeled with gidx (global annotation index from the checklist).
After reviewing, report the gidx values to DELETE.

Output: review_suspicious/page_XX.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

# -------- EDIT THESE --------
DATA_BASE_DIR = "<PATH_TO>/lung_project/data"
CHECKLIST_CSV = "<PATH_TO>/lung_project/suspicious_checklist.csv"
OUTPUT_DIR    = "<PATH_TO>/lung_project/review_suspicious"
CROP_HALF     = 70   # voxels around point; increase to see more context
ROWS_PER_PAGE = 5
WIN_MIN, WIN_MAX = -1000, 300   # display window (HU)
# ----------------------------


def world_to_voxel_lps(world_xyz, affine):
    """Annotation coords are LPS; image (RAS-oriented) affine is RAS.
    Flip X,Y sign to convert LPS->RAS before applying inverse affine."""
    w = [-world_xyz[0], -world_xyz[1], world_xyz[2]]
    inv = np.linalg.inv(affine)
    return (inv @ np.array([w[0], w[1], w[2], 1.0]))[:3]


def crop_centered(plane, cx, cy, half):
    H, W = plane.shape
    out = np.full((2*half, 2*half), plane.min(), dtype=plane.dtype)
    x0, x1, y0, y1 = int(cx-half), int(cx+half), int(cy-half), int(cy+half)
    sx0, sx1, sy0, sy1 = max(0,x0), min(W,x1), max(0,y0), min(H,y1)
    out[sy0-y0:sy0-y0+(sy1-sy0), sx0-x0:sx0-x0+(sx1-sx0)] = plane[sy0:sy1, sx0:sx1]
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CHECKLIST_CSV)
    records = df.to_dict("records")

    loader = Compose([
        LoadImaged(keys=["image"], image_only=False, meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
    ])
    cache = {}

    def get_vol(sid):
        if sid not in cache:
            data = loader({"image": f"{DATA_BASE_DIR}/imagesTr/{sid}.nii.gz"})
            cache[sid] = (data["image"][0].numpy(), np.array(data["image"].meta["affine"]))
            if len(cache) > 3:
                old = next(k for k in cache if k != sid)
                del cache[old]
        return cache[sid]

    n = len(records)
    pages = math.ceil(n / ROWS_PER_PAGE)
    print(f"Reviewing {n} suspicious annotations -> {pages} pages\n")

    for pg in range(pages):
        page = records[pg*ROWS_PER_PAGE:(pg+1)*ROWS_PER_PAGE]
        nrows = len(page)
        fig, axes = plt.subplots(nrows, 3, figsize=(12, 4*nrows))
        if nrows == 1:
            axes = axes.reshape(1, 3)

        for ri, rec in enumerate(page):
            gidx = rec["gidx"]
            sid = rec["seriesuid"]
            world = [rec["coordX"], rec["coordY"], rec["coordZ"]]
            diam = rec["diameter_mm"]
            try:
                vol, affine = get_vol(sid)
                vx, vy, vz = [int(round(v)) for v in world_to_voxel_lps(world, affine)]
                vx = max(0, min(vol.shape[0]-1, vx))
                vy = max(0, min(vol.shape[1]-1, vy))
                vz = max(0, min(vol.shape[2]-1, vz))

                axial = crop_centered(vol[:, :, vz].T, vx, vy, CROP_HALF)
                coronal = crop_centered(vol[:, vy, :].T, vx, vz, CROP_HALF)
                sagittal = crop_centered(vol[vx, :, :].T, vy, vz, CROP_HALF)

                for ci, (pl, nm) in enumerate(
                    [(axial, "Axial"), (coronal, "Coronal"), (sagittal, "Sagittal")]
                ):
                    ax = axes[ri, ci]
                    ax.imshow(pl, cmap="gray", origin="lower", vmin=WIN_MIN, vmax=WIN_MAX)
                    ax.axhline(CROP_HALF, color="red", lw=0.6, alpha=0.7)
                    ax.axvline(CROP_HALF, color="red", lw=0.6, alpha=0.7)
                    if ci == 0:
                        ax.set_ylabel(f"gidx={gidx}\n{sid[:20]}\nd={diam:.1f}mm",
                                      fontsize=8, rotation=0, ha="right", va="center",
                                      labelpad=42)
                    ax.set_title(nm, fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])
            except Exception as e:
                for ci in range(3):
                    axes[ri, ci].text(0.5, 0.5, f"gidx={gidx} ERR\n{str(e)[:30]}",
                                      ha="center", va="center", fontsize=7)
                    axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])

        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"page_{pg+1:02d}.png")
        plt.savefig(out, dpi=85, bbox_inches="tight")
        plt.close()
        print(f"  page {pg+1}/{pages} -> {os.path.basename(out)}")

    print(f"\nDone. Sheets in: {OUTPUT_DIR}")
    print("KEEP: red crosshair on a round, isolated nodule.")
    print("DELETE: crosshair on vessel/mediastinum/heart/diaphragm/liver/chest wall.")
    print("Report the gidx values to DELETE.")


if __name__ == "__main__":
    main()
