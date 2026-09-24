"""
extract_features.py — Stage 1 of the transfer-learning pipeline.

The four pretrained backbones (VGG16 / ResNet50 / MobileNetV2 / EfficientNet-B0)
are FROZEN and used as fixed feature extractors:

    image --(eval transform)-->   backbone GAP feature   (clean features)
    image --(augment transform)--> backbone GAP feature   (augmented copies)

For every image we store:
    * 1 clean feature vector  (eval transform, deterministic)
    * NUM_AUG_COPIES augmented feature vectors (random augmentation passes)
      -> this is how DATA AUGMENTATION reaches the frozen-backbone models:
         the classifier heads are later trained on the union of clean +
         augmented features.

RESUMABLE CACHING (CPU sandbox friendly):
    results/features/labels.npy            (N,)      dataset labels
    results/features/classes.json                    class names
    results/features/{model}_eval.npy      (N, D)    clean features
    results/features/{model}_aug{c}.npy    (N, D)    augmented copy c
Each (model, pass) combination is cached separately, so a long run can be
interrupted and resumed without recomputing finished passes.

CLI:
    python extract_features.py                        # everything missing
    python extract_features.py --model ResNet50       # one model
    python extract_features.py --model VGG16 --pass eval
    python extract_features.py --model VGG16 --pass aug --copy 1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (RESULTS_DIR, FEATURES_DIR, BATCH_EXTRACT, NUM_AUG_COPIES,
                    IMG_SIZE, MODEL_ORDER)
from data import (list_images, get_eval_transform, get_augment_transform,
                  ImageFolderDataset, set_seed)
from models import build_backbone

TRANSFER_MODELS = [m for m in MODEL_ORDER if m != "SimpleCNN"]


def _pass_path(model: str, kind: str, copy: int = 0) -> str:
    return os.path.join(FEATURES_DIR, f"{model}_{kind}{copy if kind == 'aug' else ''}.npy")


def extract_one_pass(model_name: str, kind: str, copy: int,
                     paths, labels, transform) -> np.ndarray:
    """Run one full pass of the dataset through the frozen backbone."""
    backbone, feat_dim = build_backbone(model_name)
    ds = ImageFolderDataset(paths, labels, transform)
    loader = DataLoader(ds, batch_size=BATCH_EXTRACT, num_workers=2)
    feats = []
    t0 = time.time()
    with torch.no_grad():
        done = 0
        for x, _ in loader:
            f = backbone(x)
            f = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f, 1), 1)
            feats.append(f.numpy())
            done += x.size(0)
            if done % (BATCH_EXTRACT * 10) == 0:
                rate = done / (time.time() - t0)
                print(f"    {model_name}/{kind}{copy}: {done}/{len(ds)} "
                      f"({rate:.1f} img/s)", flush=True)
    out = np.concatenate(feats, axis=0).astype(np.float32)
    del backbone
    return out


def ensure_labels(paths, labels, classes):
    os.makedirs(FEATURES_DIR, exist_ok=True)
    lab_path = os.path.join(FEATURES_DIR, "labels.npy")
    cls_path = os.path.join(FEATURES_DIR, "classes.json")
    if not os.path.exists(lab_path):
        np.save(lab_path, np.asarray(labels))
        with open(cls_path, "w") as f:
            json.dump(classes, f)
        print(f"[labels] saved {len(labels)} labels, classes={classes}")


def run(model_filter=None, kind="all", copy=None):
    set_seed()
    paths, labels, classes = list_images()
    print(f"Dataset: {len(paths)} images | classes = {classes}")
    ensure_labels(paths, labels, classes)

    models = [model_filter] if model_filter else TRANSFER_MODELS
    for model_name in models:
        if model_name not in TRANSFER_MODELS:
            continue
        jobs = []
        if kind in ("all", "eval"):
            jobs.append(("eval", 0))
        if kind in ("all", "aug"):
            copies = [copy] if copy is not None else range(NUM_AUG_COPIES)
            jobs += [("aug", c) for c in copies]

        for k, c in jobs:
            out_path = _pass_path(model_name, k, c)
            if os.path.exists(out_path):
                print(f"[skip] {model_name} {k}{c}: cached -> {out_path}")
                continue
            print(f"[extract] {model_name} pass={k} copy={c} ...", flush=True)
            transform = (get_eval_transform(IMG_SIZE) if k == "eval"
                         else get_augment_transform(IMG_SIZE))
            if k == "aug":
                set_seed(1000 + c)                 # different aug per copy
            arr = extract_one_pass(model_name, k, c, paths, labels, transform)
            np.save(out_path, arr)
            print(f"[done] {model_name} {k}{c}: {arr.shape} -> {out_path}")
    print("Feature extraction step complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="only this model")
    ap.add_argument("--pass_", dest="kind", default="all",
                    choices=["all", "eval", "aug"])
    ap.add_argument("--copy", type=int, default=None,
                    help="only this augmented copy index")
    args = ap.parse_args()
    run(args.model, args.kind, args.copy)


if __name__ == "__main__":
    main()
