"""
make_summary.py — publish web assets for the Next.js showcase page.

* copies figures/  -> /home/z/my-project/public/cv/figures/
* copies report/code_shots -> /home/z/my-project/public/cv/code/
* copies results/comparison_table.csv -> public/cv/
* writes public/cv/results_summary.json (contract of the showcase page)
* copies the final PDF report (if present) -> public/cv/FlowerNet_Report.pdf
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_DIR, RESULTS_DIR, FIGURES_DIR  # noqa: E402

PUBLIC_CV = "/home/z/my-project/public/cv"
MODEL_ORDER = ["SimpleCNN", "VGG16", "ResNet50", "MobileNetV2", "EfficientNet-B0"]


def main():
    os.makedirs(os.path.join(PUBLIC_CV, "figures"), exist_ok=True)
    os.makedirs(os.path.join(PUBLIC_CV, "code"), exist_ok=True)

    # ---- figures ----
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(FIGURES_DIR, f),
                         os.path.join(PUBLIC_CV, "figures", f))
    n_fig = len(os.listdir(os.path.join(PUBLIC_CV, "figures")))

    # ---- code screenshots ----
    code_src = os.path.join(PROJECT_DIR, "report", "code_shots")
    for f in sorted(os.listdir(code_src)):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(code_src, f),
                         os.path.join(PUBLIC_CV, "code", f))
    n_code = len(os.listdir(os.path.join(PUBLIC_CV, "code")))

    # ---- csv ----
    shutil.copy2(os.path.join(RESULTS_DIR, "comparison_table.csv"),
                 os.path.join(PUBLIC_CV, "comparison_table.csv"))

    # ---- results summary json ----
    with open(os.path.join(RESULTS_DIR, "metrics.json")) as f:
        res = json.load(f)

    summary = {
        "dataset": res["dataset"],
        "generated_from": "results/metrics.json (5-fold stratified CV, seed 42)",
        "models": [],
    }
    for m in MODEL_ORDER:
        mm = res["models"][m]
        summary["models"].append({
            "name": m,
            "type": mm["type"],
            "desc": mm["desc"],
            "params_total": mm["params_total"],
            "params_trainable": mm["params_trainable"],
            "aggregate": mm["aggregate"],
            "fold_val_accs": mm["fold_val_accs"],
            "train_seconds": mm["train_seconds"],
        })

    out = os.path.join(PUBLIC_CV, "results_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)

    # ---- pdf report (optional) ----
    pdf = os.path.join(PROJECT_DIR, "report", "FlowerNet_Report.pdf")
    if os.path.exists(pdf):
        shutil.copy2(pdf, os.path.join(PUBLIC_CV, "FlowerNet_Report.pdf"))
        print("[pdf] copied FlowerNet_Report.pdf")
    else:
        print("[pdf] report PDF not generated yet (skipped)")

    print(f"[done] {n_fig} figures, {n_code} code shots -> {PUBLIC_CV}")
    print(f"[done] results_summary.json -> {out}")


if __name__ == "__main__":
    main()
