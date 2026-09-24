"""
config.py — Central configuration for the FlowerNet project.

All hyper-parameters, paths and constants live here so that every other
module stays clean and the whole experiment is reproducible from one file.
"""
import os

# ---------------------------------------------------------------- paths ----
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "dataset_raw")     # downloaded raw images
DATA_DIR = os.path.join(PROJECT_DIR, "dataset")             # final 224x224 images
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
FEATURES_DIR = os.path.join(RESULTS_DIR, "features")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

# ------------------------------------------------------------ data/image ----
IMG_SIZE = 224            # input resolution for ImageNet-pretrained backbones
CNN_INPUT_SIZE = 128      # input resolution for the from-scratch SimpleCNN
NUM_AUG_COPIES = 2        # augmented feature copies per image (transfer models)
MIN_DIM = 160             # discard downloaded images smaller than this

# ------------------------------------------------- cross-validation setup ----
N_SPLITS = 5              # 5-fold stratified cross-validation
SEED = 42                 # global random seed (reproducibility)

# --------------------------------------------- frozen-backbone extraction ----
BATCH_EXTRACT = 12        # batch size for backbone feature extraction (RAM-safe)

# ------------------------------------------- classifier-head hyperparams ----
HEAD_EPOCHS = 30          # epochs for the small MLP heads on frozen features
HEAD_LR = 1e-3
HEAD_BATCH = 64
HEAD_HIDDEN = 128

# ------------------------------------------------- SimpleCNN hyperparams ----
CNN_EPOCHS = 12
CNN_BATCH = 32
CNN_LR = 1e-3

# ------------------------------------------------ learning-curve settings ----
SIZE_FRACTIONS = [0.25, 0.50, 0.75, 1.00]   # training-set-size learning curve

# --------------------------------------------------- ImageNet statistics ----
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ------------------------------------------------------ figure aesthetics ----
MODEL_COLORS = {
    "SimpleCNN":       "#7f7f7f",   # grey   — from-scratch baseline
    "VGG16":           "#d62728",   # red    — classic 2014 architecture
    "ResNet50":        "#2ca02c",   # green  — residual connections
    "MobileNetV2":     "#ff7f0e",   # orange — lightweight mobile design
    "EfficientNet-B0": "#9467bd",   # purple — compound-scaled design
}

MODEL_ORDER = ["SimpleCNN", "VGG16", "ResNet50", "MobileNetV2", "EfficientNet-B0"]
