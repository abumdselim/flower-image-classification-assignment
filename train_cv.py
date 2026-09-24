"""
train_cv.py — Stage 2: training + evaluation of all 5 models under
5-fold STRATIFIED cross-validation, plus learning-curve data collection.

For every model the script produces:
  * per-fold metrics  (accuracy / macro precision / macro recall / macro F1 /
                       macro ROC-AUC) computed on the held-out fold
  * out-of-fold (OOF) probabilities for every image in the dataset
    (each image is predicted by a model that never trained on it)
  * epoch-wise train vs validation learning curves
  * a training-set-size learning curve (accuracy vs fraction of training data)
    used to analyse under/over-fitting and the value of more data

Everything is stored in results/metrics.json.

RESUMABLE EXECUTION (CPU sandbox friendly — each CLI call fits in a small
time budget and results are merged from fragment files in results/parts/):

    python train_cv.py --model VGG16 --stage cv                  # all 5 folds
    python train_cv.py --model SimpleCNN --stage cv --folds 0,1,2
    python train_cv.py --model ResNet50 --stage size             # all fractions
    python train_cv.py --model SimpleCNN --stage size --fractions 0,1
    python train_cv.py --assemble        # merge parts -> results/metrics.json
    python train_cv.py --all-resumable   # run everything that is missing
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from config import (RESULTS_DIR, FEATURES_DIR, IMG_SIZE, CNN_INPUT_SIZE,
                    N_SPLITS, SEED, NUM_AUG_COPIES, HEAD_EPOCHS, HEAD_LR,
                    HEAD_BATCH, CNN_EPOCHS, CNN_BATCH, CNN_LR, MODEL_ORDER,
                    SIZE_FRACTIONS)
from data import (list_images, set_seed, get_eval_transform,
                  get_augment_transform, ImageFolderDataset, FeatureDataset)
from models import SimpleCNN, build_head, build_backbone, MODEL_REGISTRY

PARTS_DIR = os.path.join(RESULTS_DIR, "parts")
TRANSFER_MODELS = [m for m in MODEL_ORDER if m != "SimpleCNN"]


# =====================================================================
#  Feature loading (from the per-pass .npy cache of extract_features)
# =====================================================================
def load_features(model_name, y):
    X_eval = np.load(os.path.join(FEATURES_DIR, f"{model_name}_eval.npy"))
    aug = []
    c = 0
    while os.path.exists(os.path.join(FEATURES_DIR, f"{model_name}_aug{c}.npy")):
        aug.append(np.load(os.path.join(FEATURES_DIR, f"{model_name}_aug{c}.npy")))
        c += 1
    X_aug = np.concatenate(aug, axis=0)
    A = len(aug)
    assert X_eval.shape[0] == len(y), "feature rows != dataset size"
    return X_eval, X_aug, A


# =====================================================================
#  Generic training loop shared by the MLP heads and the SimpleCNN
# =====================================================================
def train_loop(model, train_loader, val_loader, device, epochs, lr,
               n_classes, oof_probs=None, oof_idx=None):
    """
    Train `model`, track learning curves, keep the best-validation-accuracy
    weights and finally return metrics + curves + OOF probabilities.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    curves = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, best_train = -1.0, None, 0.0

    for epoch in range(epochs):
        # ---------------- train ----------------
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * y.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_n += y.size(0)
        train_loss, train_acc = tr_loss / tr_n, tr_correct / tr_n

        # ---------------- validate ----------------
        model.eval()
        va_loss, va_correct, va_n = 0.0, 0, 0
        probs_all = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item() * y.size(0)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_n += y.size(0)
                probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
        val_loss, val_acc = va_loss / va_n, va_correct / va_n

        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)
        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)

        if val_acc > best_val:                       # keep best model state
            best_val, best_train = val_acc, train_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    # restore best weights -> OOF probabilities on the held-out fold
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probs_list = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            probs_list.append(torch.softmax(model(x), dim=1).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    if oof_probs is not None and oof_idx is not None:
        oof_probs[oof_idx] = probs

    return {"curves": curves, "best_val_acc": best_val,
            "train_acc_at_best": best_train, "val_probs": probs}


def evaluate_metrics(y_true, y_pred, y_prob):
    """The evaluation metrics used to compare all models."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred,
                                                 average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred,
                                              average="macro", zero_division=0)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0)),
        "auc_macro": float(roc_auc_score(y_true, y_prob,
                                         multi_class="ovr", average="macro")),
    }


# =====================================================================
#  Model-specific fold training
# =====================================================================
def train_fold_transfer(X_eval, X_aug, A, y, tr_idx, va_idx,
                        device, n_classes, epochs=HEAD_EPOCHS):
    """Train the MLP head on frozen features for one CV fold."""
    N = len(y)
    # augmented rows of the training originals: copy c stores rows [c*N, (c+1)*N)
    aug_rows = np.concatenate([np.arange(N)[tr_idx] + c * N for c in range(A)])
    X_tr = np.vstack([X_eval[tr_idx], X_aug[aug_rows]])
    # labels: originals once + every augmented copy (same order as the rows)
    y_tr = np.concatenate([y[tr_idx]] * (A + 1))

    head = build_head(X_eval.shape[1], n_classes).to(device)
    train_loader = DataLoader(FeatureDataset(X_tr, y_tr), batch_size=HEAD_BATCH,
                              shuffle=True)
    val_loader = DataLoader(FeatureDataset(X_eval[va_idx], y[va_idx]),
                            batch_size=HEAD_BATCH, shuffle=False)
    return train_loop(head, train_loader, val_loader, device, epochs, HEAD_LR,
                      n_classes)


def train_fold_cnn(paths, labels, tr_idx, va_idx, device, n_classes,
                   epochs=CNN_EPOCHS):
    """Train the from-scratch SimpleCNN on pixels for one CV fold."""
    cnn = SimpleCNN(n_classes).to(device)
    train_ds = ImageFolderDataset([paths[i] for i in tr_idx],
                                  [labels[i] for i in tr_idx],
                                  get_augment_transform(CNN_INPUT_SIZE))
    val_ds = ImageFolderDataset([paths[i] for i in va_idx],
                                [labels[i] for i in va_idx],
                                get_eval_transform(CNN_INPUT_SIZE))
    train_loader = DataLoader(train_ds, batch_size=CNN_BATCH, shuffle=True,
                              num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CNN_BATCH, shuffle=False,
                            num_workers=2)
    return train_loop(cnn, train_loader, val_loader, device, epochs, CNN_LR,
                      n_classes)


# =====================================================================
#  Learning curve vs training-set size
# =====================================================================
def stratified_subset(y_tr, frac, seed):
    """Pick a stratified subset of the training fold (per-class rounding)."""
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y_tr):
        idx_c = np.where(y_tr == c)[0]
        rng.shuffle(idx_c)
        keep.extend(idx_c[:max(1, int(round(frac * len(idx_c))))])
    return np.sort(np.array(keep))


# =====================================================================
#  Fragment helpers
# =====================================================================
def part_path(model_name, stage):
    return os.path.join(PARTS_DIR, f"{model_name}__{stage}.json")


def load_part(model_name, stage):
    p = part_path(model_name, stage)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    if stage == "cv":
        return {"folds": {}}
    return {"fractions": {}}


def save_part(model_name, stage, part):
    os.makedirs(PARTS_DIR, exist_ok=True)
    with open(part_path(model_name, stage), "w") as f:
        json.dump(part, f)


def get_folds(y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


# =====================================================================
#  Stage: cross-validation (per model, per fold-group)
# =====================================================================
def stage_cv(model_name, folds_sel=None):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels, classes = list_images()
    y = np.asarray(labels)
    n_classes = len(classes)
    info = MODEL_REGISTRY[model_name]

    part = load_part(model_name, "cv")
    folds = get_folds(y)
    todo = [k for k in range(N_SPLITS)
            if (folds_sel is None or k in folds_sel) and str(k) not in part["folds"]]
    if not todo:
        print(f"[cv] {model_name}: all folds already cached")
        return

    t0 = time.time()
    X_eval = X_aug = None
    A = NUM_AUG_COPIES
    if info["type"] == "transfer":
        X_eval, X_aug, A = load_features(model_name, y)

    for k in todo:
        tr_idx, va_idx = folds[k]
        set_seed(SEED + k)
        if info["type"] == "transfer":
            r = train_fold_transfer(X_eval, X_aug, A, y, tr_idx, va_idx,
                                    device, n_classes)
        else:
            r = train_fold_cnn(paths, labels, tr_idx, va_idx, device, n_classes)
        y_pred = r["val_probs"].argmax(1)
        m = evaluate_metrics(y[va_idx], y_pred, r["val_probs"])
        m.update({"fold": k + 1, "train_acc_at_best": r["train_acc_at_best"],
                  "best_val_acc": r["best_val_acc"]})
        part["folds"][str(k)] = {
            "metrics": m,
            "curves": r["curves"],
            "va_idx": va_idx.tolist(),
            "probs": r["val_probs"].astype(float).tolist(),
        }
        save_part(model_name, "cv", part)          # checkpoint after every fold
        print(f"[cv] {model_name} fold {k + 1}/{N_SPLITS}: "
              f"acc={m['accuracy']:.3f} f1={m['f1_macro']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    part["elapsed"] = part.get("elapsed", 0.0) + (time.time() - t0)
    save_part(model_name, "cv", part)
    print(f"[cv] {model_name}: {len(part['folds'])}/{N_SPLITS} folds done")


# =====================================================================
#  Stage: training-set-size learning curve (per model, per fraction)
# =====================================================================
def stage_size(model_name, frac_sel=None):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels, classes = list_images()
    y = np.asarray(labels)
    n_classes = len(classes)
    info = MODEL_REGISTRY[model_name]

    part = load_part(model_name, "size")
    folds = get_folds(y)
    todo = []
    for fi, frac in enumerate(SIZE_FRACTIONS):
        if frac_sel is not None and fi not in frac_sel:
            continue
        if str(fi) not in part["fractions"]:
            todo.append((fi, frac))
    if not todo:
        print(f"[size] {model_name}: all fractions already cached")
        return

    t0 = time.time()
    X_eval = X_aug = None
    A = NUM_AUG_COPIES
    if info["type"] == "transfer":
        X_eval, X_aug, A = load_features(model_name, y)
        N = len(y)

    for fi, frac in todo:
        tr_accs, va_accs, ns = [], [], []
        for k, (tr_idx, va_idx) in enumerate(folds):
            sub = stratified_subset(y[tr_idx], frac, SEED * 100 + k)
            if info["type"] == "transfer":
                sel_orig = tr_idx[sub]
                aug_rows = np.concatenate([sel_orig + c * N for c in range(A)])
                X_tr = np.vstack([X_eval[sel_orig], X_aug[aug_rows]])
                y_tr = np.concatenate([y[sel_orig]] * (A + 1))
                head = build_head(X_eval.shape[1], n_classes).to(device)
                tl = DataLoader(FeatureDataset(X_tr, y_tr), batch_size=HEAD_BATCH,
                                shuffle=True)
                vl = DataLoader(FeatureDataset(X_eval[va_idx], y[va_idx]),
                                batch_size=HEAD_BATCH, shuffle=False)
                r = train_loop(head, tl, vl, device, HEAD_EPOCHS, HEAD_LR,
                               n_classes)
                ns.append(len(y_tr))
            else:
                sel = tr_idx[sub]
                r = train_fold_cnn(paths, labels, sel, va_idx, device, n_classes,
                                   epochs=max(8, int(CNN_EPOCHS * frac)))
                ns.append(len(sel))
            tr_accs.append(r["train_acc_at_best"])
            va_accs.append(r["best_val_acc"])
        part["fractions"][str(fi)] = {
            "frac": frac,
            "train_acc": float(np.mean(tr_accs)),
            "val_acc": float(np.mean(va_accs)),
            "n_train": float(np.mean(ns)),
        }
        save_part(model_name, "size", part)        # checkpoint after every fraction
        print(f"[size] {model_name} frac={frac:.2f} "
              f"train={part['fractions'][str(fi)]['train_acc']:.3f} "
              f"val={part['fractions'][str(fi)]['val_acc']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    part["elapsed"] = part.get("elapsed", 0.0) + (time.time() - t0)
    save_part(model_name, "size", part)


# =====================================================================
#  Assemble all fragments into results/metrics.json
# =====================================================================
def count_params(module):
    return sum(p.numel() for p in module.parameters())


def model_param_counts(model_name, n_classes):
    info = MODEL_REGISTRY[model_name]
    if info["type"] == "transfer":
        import torchvision.models as tvm
        # architecture only (weights=None) -> no download, just param counting
        if model_name == "VGG16":
            net = tvm.vgg16(weights=None)
            backbone = net.features
        elif model_name == "ResNet50":
            net = tvm.resnet50(weights=None)
            backbone = nn.Sequential(*list(net.children())[:-1])
        elif model_name == "MobileNetV2":
            net = tvm.mobilenet_v2(weights=None)
            backbone = net.features
        else:  # EfficientNet-B0
            net = tvm.efficientnet_b0(weights=None)
            backbone = net.features
        head = build_head({"VGG16": 512, "ResNet50": 2048,
                           "MobileNetV2": 1280, "EfficientNet-B0": 1280}[model_name],
                          n_classes)
        total = count_params(backbone) + count_params(head)
        trainable = count_params(head)
        del net, backbone, head
        return total, trainable
    cnn = SimpleCNN(n_classes)
    n = count_params(cnn)
    del cnn
    return n, n


def assemble():
    paths, labels, classes = list_images()
    y = np.asarray(labels)
    n_classes = len(classes)
    counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}

    missing = []
    for m in MODEL_ORDER:
        cv_part = load_part(m, "cv")
        size_part = load_part(m, "size")
        if len(cv_part["folds"]) < N_SPLITS:
            missing.append(f"{m}:cv({len(cv_part['folds'])}/{N_SPLITS})")
        if len(size_part["fractions"]) < len(SIZE_FRACTIONS):
            missing.append(f"{m}:size({len(size_part['fractions'])}/{len(SIZE_FRACTIONS)})")
    if missing:
        print("Cannot assemble — incomplete parts: " + ", ".join(missing))
        return None

    results = {"dataset": {"classes": classes, "counts": counts,
                           "total": len(paths), "n_splits": N_SPLITS,
                           "seed": SEED}, "models": {}}
    for m in MODEL_ORDER:
        cv_part = load_part(m, "cv")
        size_part = load_part(m, "size")
        info = MODEL_REGISTRY[m]

        fold_metrics, fold_curves, fold_best = [], [], []
        oof_probs = np.zeros((len(y), n_classes), dtype=np.float32)
        for k in range(N_SPLITS):
            fd = cv_part["folds"][str(k)]
            fold_metrics.append(fd["metrics"])
            fold_curves.append(fd["curves"])
            fold_best.append(fd["metrics"]["best_val_acc"])
            oof_probs[fd["va_idx"]] = np.asarray(fd["probs"], dtype=np.float32)

        agg = {}
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "auc_macro", "train_acc_at_best", "best_val_acc"]:
            vals = np.array([fm[key] for fm in fold_metrics], dtype=float)
            agg[f"{key}_mean"] = float(vals.mean())
            agg[f"{key}_std"] = float(vals.std())
        mean_curves = {k: list(np.mean([c[k] for c in fold_curves], axis=0))
                       for k in fold_curves[0]}

        fractions = [size_part["fractions"][str(i)] for i in range(len(SIZE_FRACTIONS))]
        size_curve = {"fractions": SIZE_FRACTIONS,
                      "train_acc": [f["train_acc"] for f in fractions],
                      "val_acc": [f["val_acc"] for f in fractions],
                      "n_train": [f["n_train"] for f in fractions]}

        total, trainable = model_param_counts(m, n_classes)
        results["models"][m] = {
            "type": info["type"], "desc": info["desc"],
            "params_total": total, "params_trainable": trainable,
            "fold_metrics": fold_metrics, "aggregate": agg,
            "mean_epoch_curves": mean_curves,
            "fold_val_accs": [fm["accuracy"] for fm in fold_metrics],
            "size_curve": size_curve,
            "oof_probs": oof_probs.tolist(), "oof_y": y.tolist(),
            "train_seconds": round(cv_part.get("elapsed", 0.0)
                                   + size_part.get("elapsed", 0.0), 1),
        }
        print(f"[assemble] {m}: acc={agg['accuracy_mean']:.4f}"
              f"+/-{agg['accuracy_std']:.4f} f1={agg['f1_macro_mean']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved assembled results -> {out}")
    return results


# =====================================================================
#  Convenience: run whatever is missing (one small step at a time)
# =====================================================================
def next_missing_job():
    """Return the next (model, stage, selector) that has not been computed."""
    paths, labels, classes = list_images()
    y = np.asarray(labels)
    for m in TRANSFER_MODELS:
        cv_part = load_part(m, "cv")
        todo = [k for k in range(N_SPLITS) if str(k) not in cv_part["folds"]]
        if todo:
            return (m, "cv", None)
        size_part = load_part(m, "size")
        todo = [i for i in range(len(SIZE_FRACTIONS))
                if str(i) not in size_part["fractions"]]
        if todo:
            return (m, "size", None)
    for m in ["SimpleCNN"]:
        cv_part = load_part(m, "cv")
        todo = [k for k in range(N_SPLITS) if str(k) not in cv_part["folds"]]
        if todo:
            return (m, "cv", todo[:1])            # one fold per call
        size_part = load_part(m, "size")
        todo = [i for i in range(len(SIZE_FRACTIONS))
                if str(i) not in size_part["fractions"]]
        if todo:
            return (m, "size", todo[:1])          # one fraction per call
    return None


def run_next():
    job = next_missing_job()
    if job is None:
        print("[all] nothing missing — run --assemble")
        return False
    model, stage, sel = job
    print(f"[next] {model} / {stage} / sel={sel}")
    if stage == "cv":
        stage_cv(model, sel)
    else:
        stage_size(model, sel)
    return True


def run_all_resumable(max_seconds=480):
    """Keep executing missing jobs until the time budget is exhausted."""
    t0 = time.time()
    while time.time() - t0 < max_seconds:
        if not run_next():
            break
    assemble_if_ready()


def assemble_if_ready():
    paths, labels, classes = list_images()
    y = np.asarray(labels)
    ready = True
    for m in MODEL_ORDER:
        cv_part = load_part(m, "cv")
        size_part = load_part(m, "size")
        if len(cv_part["folds"]) < N_SPLITS or \
                len(size_part["fractions"]) < len(SIZE_FRACTIONS):
            ready = False
    if ready:
        assemble()
    else:
        print("[assemble] skipped — parts still incomplete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--stage", default=None, choices=["cv", "size"])
    ap.add_argument("--folds", default=None, help="comma list of fold indices")
    ap.add_argument("--fractions", default=None, help="comma list of fraction idx")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--next", action="store_true", help="run one missing job")
    ap.add_argument("--all-resumable", action="store_true")
    args = ap.parse_args()

    if args.assemble:
        assemble()
    elif args.next:
        run_next()
    elif args.all_resumable:
        run_all_resumable()
    elif args.model and args.stage:
        folds_sel = ([int(x) for x in args.folds.split(",")]
                     if args.folds else None)
        frac_sel = ([int(x) for x in args.fractions.split(",")]
                    if args.fractions else None)
        if args.stage == "cv":
            stage_cv(args.model, folds_sel)
        else:
            stage_size(args.model, frac_sel)
    else:
        ap.print_help()
