"""
evaluate.py — Stage 3: aggregate results, build every evaluation figure and
the final comparison table.

Produces (in figures/):
  01_class_distribution.png        dataset balance check
  02_sample_grid.png               sample images per class
  03_augmentation_showcase.png     what the augmentation pipeline does
  04_cv_accuracy_bars.png          mean 5-fold CV accuracy +/- std per model
  05_metrics_comparison.png        accuracy / precision / recall / F1 / AUC
  06_fold_stability.png            per-fold accuracy of every model
  07..11_curves_<model>.png        epoch learning curves (overfitting check)
  12_size_curve_all.png            accuracy vs training-set size
  13_overfit_gap.png               final train-val accuracy gap per model
  14_confusion_<model>.png         out-of-fold confusion matrices
  15_roc_macro_all.png             macro one-vs-rest ROC curves
  16_summary_heatmap.png           metric heatmap of all models
and results/comparison_table.csv.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import (RESULTS_DIR, FIGURES_DIR, DATA_DIR, MODEL_ORDER,
                    MODEL_COLORS, N_SPLITS)
from data import list_images, get_augment_transform, denormalize
import torch

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})
DPI = 150


# ------------------------------------------------------------ helpers ----
def savefig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}")


def load_results():
    with open(os.path.join(RESULTS_DIR, "metrics.json")) as f:
        return json.load(f)


# ------------------------------------------------------- dataset plots ----
def fig_class_distribution(res):
    classes, counts = res["dataset"]["classes"], res["dataset"]["counts"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(classes, [counts[c] for c in classes],
                  color=["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"][:len(classes)])
    for b, c in zip(bars, classes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                str(counts[c]), ha="center", fontweight="bold")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Custom dataset — {res['dataset']['total']} images, "
                 f"{len(classes)} classes (per-class counts)")
    savefig(fig, "01_class_distribution.png")


def fig_sample_grid(classes):
    import random
    random.seed(7)
    n_show = 5
    fig, axes = plt.subplots(len(classes), n_show,
                             figsize=(n_show * 2.0, len(classes) * 2.0))
    for r, cls in enumerate(classes):
        cdir = os.path.join(DATA_DIR, cls)
        files = sorted(os.listdir(cdir))
        picks = random.sample(files, n_show)
        for c in range(n_show):
            ax = axes[r, c]
            ax.imshow(Image.open(os.path.join(cdir, picks[c])))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(cls, fontsize=11, fontweight="bold")
            if r == 0:
                ax.set_title(picks[c][:18], fontsize=7)
    fig.suptitle("Sample images of the custom dataset (one row per class)",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "02_sample_grid.png")


def fig_augmentation_showcase(classes):
    from data import load_pil
    aug = get_augment_transform(224)
    picks = []
    for cls in classes[:2]:
        cdir = os.path.join(DATA_DIR, cls)
        picks.append(os.path.join(cdir, sorted(os.listdir(cdir))[0]))
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for r, p in enumerate(picks):
        img = load_pil(p)
        axes[r, 0].imshow(img); axes[r, 0].set_title("original", fontsize=9)
        for c in range(1, 4):
            t = aug(img)                       # random augmented view
            arr = denormalize(t).permute(1, 2, 0).numpy().clip(0, 1)
            axes[r, c].imshow(arr)
            axes[r, c].set_title(f"augmented #{c}", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Data augmentation — random crop / flip / rotation / colour jitter "
                 "(training data only)", fontsize=12)
    fig.tight_layout()
    savefig(fig, "03_augmentation_showcase.png")


# ------------------------------------------------------ model results ----
def fig_cv_accuracy_bars(res):
    models = MODEL_ORDER
    means = [res["models"][m]["aggregate"]["accuracy_mean"] for m in models]
    stds = [res["models"][m]["aggregate"]["accuracy_std"] for m in models]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(models, means, yerr=stds, capsize=5,
                  color=[MODEL_COLORS[m] for m in models], width=0.62)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + s + 0.012,
                f"{m:.3f}", ha="center", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{N_SPLITS}-fold stratified cross-validation accuracy "
                 "(mean ± std)")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    savefig(fig, "04_cv_accuracy_bars.png")


def fig_metrics_comparison(res):
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"),
               ("recall_macro", "Recall (macro)"), ("f1_macro", "F1 (macro)"),
               ("auc_macro", "ROC-AUC (macro)")]
    x = np.arange(len(metrics))
    w = 0.16
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, m in enumerate(MODEL_ORDER):
        vals = [res["models"][m]["aggregate"][f"{k}_mean"] for k, _ in metrics]
        ax.bar(x + (i - 2) * w, vals, w, label=m, color=MODEL_COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (mean over 5 folds)")
    ax.set_title("Evaluation-metric comparison across the five models")
    ax.legend(ncol=3, fontsize=8.5, frameon=False)
    savefig(fig, "05_metrics_comparison.png")


def fig_fold_stability(res):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m in MODEL_ORDER:
        accs = res["models"][m]["fold_val_accs"]
        ax.plot(range(1, N_SPLITS + 1), accs, "o--", label=m,
                color=MODEL_COLORS[m])
    ax.set_xticks(range(1, N_SPLITS + 1))
    ax.set_xlabel("Fold"); ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Per-fold validation accuracy (stability across folds)")
    ax.legend(fontsize=8.5)
    savefig(fig, "06_fold_stability.png")


def fig_epoch_curves(res, model_name):
    """Learning curves over epochs — the overfitting diagnostic."""
    c = res["models"][model_name]["mean_epoch_curves"]
    ep = np.arange(1, len(c["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    col = MODEL_COLORS[model_name]
    axes[0].plot(ep, c["train_loss"], color=col, label="training loss")
    axes[0].plot(ep, c["val_loss"], color=col, linestyle="--", label="validation loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title(f"{model_name} — loss"); axes[0].legend(fontsize=8.5)
    axes[1].plot(ep, c["train_acc"], color=col, label="training accuracy")
    axes[1].plot(ep, c["val_acc"], color=col, linestyle="--",
                 label="validation accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].set_title(f"{model_name} — accuracy"); axes[1].legend(fontsize=8.5)
    fig.suptitle(f"Learning curves of {model_name} "
                 "(mean over the 5 CV folds)", fontsize=11)
    fig.tight_layout()
    savefig(fig, f"07_curves_{model_name}.png".replace(" ", ""))


def fig_size_curves(res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in MODEL_ORDER:
        sc = res["models"][m]["size_curve"]
        gap = np.array(sc["train_acc"]) - np.array(sc["val_acc"])
        axes[0].plot(sc["n_train"], sc["val_acc"], "o-", label=m,
                     color=MODEL_COLORS[m])
        axes[1].plot(sc["n_train"], gap, "o--", label=m, color=MODEL_COLORS[m])
    axes[0].set_xlabel("Training images used (incl. augmented copies)")
    axes[0].set_ylabel("CV validation accuracy")
    axes[0].set_title("Learning curve: accuracy vs training-set size")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.10, color="grey", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Training images used (incl. augmented copies)")
    axes[1].set_ylabel("Train − Val accuracy gap")
    axes[1].set_title("Overfitting gap vs training-set size")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_ylim(bottom=None)
    fig.tight_layout()
    savefig(fig, "12_size_curve_all.png")


def fig_overfit_gap(res):
    models = MODEL_ORDER
    gaps = [res["models"][m]["aggregate"]["train_acc_at_best_mean"]
            - res["models"][m]["aggregate"]["best_val_acc_mean"] for m in models]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(models, gaps, color=[MODEL_COLORS[m] for m in models],
                  width=0.6)
    for b, g in zip(bars, gaps):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.004, f"{g:+.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Train accuracy − Validation accuracy")
    ax.set_title("Overfitting check: train–validation gap at the best epoch "
                 "(mean of 5 folds)")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    savefig(fig, "13_overfit_gap.png")


def plot_confusion(ax, cm, classes, title):
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right",
                  fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="black" if cm[i, j] < 0.6 else "white", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.grid(False)


def fig_confusion(res, model_name):
    y = np.array(res["models"][model_name]["oof_y"])
    probs = np.array(res["models"][model_name]["oof_probs"])
    classes = res["dataset"]["classes"]
    cm = confusion_matrix(y, probs.argmax(1), normalize="true")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    plot_confusion(ax, cm, classes,
                   f"{model_name} — out-of-fold confusion matrix "
                   "(row-normalised)")
    fig.tight_layout()
    savefig(fig, f"14_confusion_{model_name}.png".replace(" ", ""))


def fig_roc(res):
    classes = res["dataset"]["classes"]
    y_bin_all = None
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for m in MODEL_ORDER:
        y = np.array(res["models"][m]["oof_y"])
        probs = np.array(res["models"][m]["oof_probs"])
        y_bin = label_binarize(y, classes=range(len(classes)))
        # macro-average ROC: interpolate each class curve and average
        mean_tpr, mean_fpr = np.zeros(100), np.linspace(0, 1, 100)
        for c in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr /= len(classes)
        macro_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color=MODEL_COLORS[m],
                label=f"{m} (macro-AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (macro average, pooled out-of-fold "
                 "predictions)")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "15_roc_macro_all.png")


def fig_summary_heatmap(res):
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision"),
               ("recall_macro", "Recall"), ("f1_macro", "F1-score"),
               ("auc_macro", "ROC-AUC")]
    M = np.zeros((len(MODEL_ORDER), len(metrics)))
    for i, m in enumerate(MODEL_ORDER):
        for j, (k, _) in enumerate(metrics):
            M[i, j] = res["models"][m]["aggregate"][f"{k}_mean"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(M, cmap="YlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(metrics)), [l for _, l in metrics], fontsize=9)
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="black" if M[i, j] < 0.85 else "white", fontsize=9)
    ax.set_title("Summary of evaluation metrics (mean over 5 CV folds)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    savefig(fig, "16_summary_heatmap.png")


# ------------------------------------------------------------- table ----
def write_csv(res):
    path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
            "auc_macro", "train_acc_at_best", "best_val_acc"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"{k}_mean" for k in keys]
                   + [f"{k}_std" for k in keys] + ["params_total", "params_trainable"])
        for m in MODEL_ORDER:
            a = res["models"][m]["aggregate"]
            w.writerow([m] + [f"{a[k + '_mean']:.4f}" for k in keys]
                       + [f"{a[k + '_std']:.4f}" for k in keys]
                       + [res["models"][m]["params_total"],
                          res["models"][m]["params_trainable"]])
    print(f"  [csv] {path}")


def print_summary(res):
    """Console summary table — captured as a screenshot for the report."""
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
            "auc_macro"]
    print("\n" + "=" * 78)
    print(f"{'FINAL CROSS-VALIDATION RESULTS':^78}")
    print("=" * 78)
    hdr = f"{'Model':<17}" + "".join(f"{k.split('_')[0].upper():>12}" for k in keys) \
          + f"{'Acc±Std':>12}{'Gap':>8}"
    print(hdr)
    print("-" * 78)
    for m in MODEL_ORDER:
        a = res["models"][m]["aggregate"]
        gap = a["train_acc_at_best_mean"] - a["best_val_acc_mean"]
        row = f"{m:<17}"
        for k in keys:
            row += f"{a[k + '_mean']:>12.4f}"
        row += f"{a['accuracy_mean']:>8.3f}±{a['accuracy_std']:.3f}"
        row += f"{gap:>+8.3f}"
        print(row)
    print("-" * 78)
    print("Gap = train accuracy − validation accuracy at the best epoch "
          "(overfitting indicator)")
    best = max(MODEL_ORDER, key=lambda m: res["models"][m]["aggregate"]["f1_macro_mean"])
    print(f"BEST MODEL (macro-F1): {best} — "
          f"F1 = {res['models'][best]['aggregate']['f1_macro_mean']:.4f}, "
          f"accuracy = {res['models'][best]['aggregate']['accuracy_mean']:.4f}")
    print("=" * 78)


# --------------------------------------------------------------- main ----
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    res = load_results()
    print("Generating figures ...")
    fig_class_distribution(res)
    fig_sample_grid(res["dataset"]["classes"])
    fig_augmentation_showcase(res["dataset"]["classes"])
    fig_cv_accuracy_bars(res)
    fig_metrics_comparison(res)
    fig_fold_stability(res)
    for m in MODEL_ORDER:
        fig_epoch_curves(res, m)
        fig_confusion(res, m)
    fig_size_curves(res)
    fig_overfit_gap(res)
    fig_roc(res)
    fig_summary_heatmap(res)
    write_csv(res)
    print_summary(res)


if __name__ == "__main__":
    main()
