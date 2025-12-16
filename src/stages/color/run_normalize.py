# src/stages/color/run_normalize.py
import os, sys, csv
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # .../palmvision
CLEAN_DIR    = PROJECT_ROOT / "data" / "clean"
NORM_DIR     = PROJECT_ROOT / "data" / "normalized"
MANIFESTS    = PROJECT_ROOT / "manifests"

CLASSES = ["Bug", "Dubas", "Healthy", "Honey"]
NORM_DIR.mkdir(parents=True, exist_ok=True)
for c in CLASSES:
    (NORM_DIR / c).mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

def normalize_one(src_path: Path, dst_path: Path):
    with Image.open(src_path) as im:
        mode_before = im.mode
        # 1) EXIF orientation transpose (deskew if device rotated)
        im = ImageOps.exif_transpose(im)
        # 2) Force RGB (drop alpha if present)
        im = im.convert("RGB")
        # Save as JPEG (consistent extension)
        dst_path = dst_path.with_suffix(".jpg")
        im.save(dst_path, format="JPEG", quality=95, optimize=True)
        w, h = im.size
        return mode_before, "RGB", w, h, dst_path

def main():
    rows = []
    kept = {c: 0 for c in CLASSES}
    total = 0

    # iterate clean set
    for c in CLASSES:
        cdir = CLEAN_DIR / c
        if not cdir.exists():
            continue
        files = [p for p in cdir.iterdir() if p.is_file()]
        for i, src in enumerate(tqdm(files, desc=f"Normalize {c:7s}", ncols=100)):
            total += 1
            dst = (NORM_DIR / c / src.name)
            try:
                mode_before, mode_after, w, h, real_dst = normalize_one(src, dst)
                rows.append({
                    "status": "normalized",
                    "class": c,
                    "src_path": str(src),
                    "dst_path": str(real_dst),
                    "mode_before": mode_before,
                    "mode_after": mode_after,
                    "width": w,
                    "height": h,
                })
                kept[c] += 1
            except Exception as e:
                rows.append({
                    "status": "failed",
                    "class": c,
                    "src_path": str(src),
                    "dst_path": "",
                    "mode_before": "",
                    "mode_after": "",
                    "width": "",
                    "height": "",
                    "error": repr(e),
                })

    # write manifest
    out = MANIFESTS / "manifest_normalized.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "status","class","src_path","dst_path",
                "mode_before","mode_after","width","height","error"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n[SUMMARY]")
    for c, n in kept.items():
        print(f"  Normalized {c:7s}: {n}")
    print(f"[INFO] Normalized manifest: {out}")

if __name__ == "__main__":
    sys.exit(main())
# =========================
# CLASS NAMES (from config)
# =========================

# This assumes your default.yaml has something like:
# classes: ["Bug", "Dubas", "Healthy", "Honey"]
_cfg = load_config()
CLASS_NAMES = list(_cfg.classes)
