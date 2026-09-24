"""
prepare_dataset.py — convert downloaded raw images into the final dataset.

For every image in dataset_raw/<class>/ :
    * open with PIL, force RGB
    * resize so the SHORT side is 256, then center-crop to 224x224
    * save as high-quality JPEG into dataset/<class>/

Result: a clean, uniform 224x224 RGB custom dataset ready for training.
"""
import os
from PIL import Image

from config import RAW_DATA_DIR, DATA_DIR

TARGET = 224
PAD = 256


def process_image(src_path: str, dst_path: str) -> bool:
    try:
        img = Image.open(src_path).convert("RGB")
        w, h = img.size
        if min(w, h) < 160:                      # too small to be useful
            return False
        scale = PAD / min(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
        w, h = img.size
        left, top = (w - TARGET) // 2, (h - TARGET) // 2
        img = img.crop((left, top, left + TARGET, top + TARGET))
        img.save(dst_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    classes = sorted(d for d in os.listdir(RAW_DATA_DIR)
                     if os.path.isdir(os.path.join(RAW_DATA_DIR, d)))
    total = 0
    print(f"Preparing final dataset at: {DATA_DIR}")
    for cls in classes:
        src_dir = os.path.join(RAW_DATA_DIR, cls)
        dst_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(dst_dir, exist_ok=True)
        kept, dropped = 0, 0
        for f in sorted(os.listdir(src_dir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            ok = process_image(os.path.join(src_dir, f),
                               os.path.join(dst_dir, f"{cls}_{kept:04d}.jpg"))
            kept += ok
            dropped += (not ok)
        total += kept
        print(f"  {cls:<12s} kept={kept:3d}  dropped={dropped}")
    print(f"TOTAL images in final dataset: {total}")


if __name__ == "__main__":
    main()
