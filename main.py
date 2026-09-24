"""
main.py — One-command driver for the whole experiment.

Pipeline:
  1) verify the custom dataset
  2) extract frozen-backbone features (VGG16 / ResNet50 / MobileNetV2 /
     EfficientNet-B0) with data augmentation (cached)
  3) train all 5 models under 5-fold stratified cross-validation
  4) build every evaluation figure + the final comparison table

Run:  python main.py
"""
import os

import config
from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR, FEATURES_DIR
from data import list_images, set_seed


def banner(text=""):
    print("\n" + "=" * 66)
    print(f"  {text:^62}")
    print("=" * 66, flush=True)


def main():
    set_seed()
    for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, FEATURES_DIR):
        os.makedirs(d, exist_ok=True)

    banner("FlowerNet — 5 CNN models x 5-fold cross-validation")

    # ---------- 1) dataset ----------
    print("[1/4] Checking custom dataset ...")
    paths, labels, classes = list_images()
    if len(paths) < 300 or len(classes) < 3:
        raise RuntimeError("dataset incomplete — run scripts/collect_dataset.py "
                           "and prepare_dataset.py first (need 300 images in 3 classes)")
    for i, c in enumerate(classes):
        print(f"    class {c:<12s}: {labels.count(i):3d} images")
    print(f"    TOTAL: {len(paths)} images (requirement: 100 per class x 3 "
          f"classes = 300) OK")

    # ---------- 2) feature extraction ----------
    print("\n[2/4] Extracting frozen-backbone features (+ augmentation) ...")
    import extract_features
    extract_features.main()

    # ---------- 3) cross-validation training ----------
    print("\n[3/4] 5-fold stratified cross-validation for 5 models ...")
    import train_cv
    train_cv.run_all_resumable(max_seconds=10 ** 9)   # run everything missing
    train_cv.assemble_if_ready()

    # ---------- 4) evaluation figures ----------
    print("\n[4/4] Building evaluation figures and tables ...")
    import evaluate
    evaluate.main()

    banner("ALL DONE — see results/ and figures/")
    print(f"  metrics json : {os.path.join(RESULTS_DIR, 'metrics.json')}")
    print(f"  csv table    : {os.path.join(RESULTS_DIR, 'comparison_table.csv')}")
    print(f"  figures      : {FIGURES_DIR} ({len(os.listdir(FIGURES_DIR))} files)")


if __name__ == "__main__":
    main()
