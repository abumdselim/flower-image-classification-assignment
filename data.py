"""
data.py — Dataset loading, transforms (including augmentation) and helpers.

The dataset is a *custom* flower dataset collected for this project:
    dataset/<class_name>/<class_name>_XXXX.jpg  (224x224 RGB JPEG)

Two transform pipelines are provided:
  * eval transform    — deterministic resize + center crop (used for validation
                        and for frozen-backbone feature extraction)
  * augment transform — random resized crop / flip / rotation / color jitter
                        (data augmentation, applied to TRAINING data only)
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (DATA_DIR, IMG_SIZE, CNN_INPUT_SIZE, IMAGENET_MEAN,
                    IMAGENET_STD, SEED, MIN_DIM)


# ------------------------------------------------------------- seeding ----
def set_seed(seed: int = SEED):
    """Seed python / numpy / torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------ dataset listing ----
def list_images(data_dir: str = DATA_DIR):
    """
    Scan the dataset folder and return (paths, labels, class_names).
    `class_names` is sorted alphabetically; label = index into class_names.
    """
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    paths, labels = [], []
    for label, cls in enumerate(classes):
        cdir = os.path.join(data_dir, cls)
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(".jpg"))
        for f in files:
            paths.append(os.path.join(cdir, f))
            labels.append(label)
    return paths, labels, classes


# ---------------------------------------------------------- transforms ----
def get_eval_transform(img_size: int = IMG_SIZE):
    """Deterministic pre-processing used for validation / feature extraction."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),      # 256 for 224, 146 for 128
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_augment_transform(img_size: int = IMG_SIZE):
    """
    DATA AUGMENTATION pipeline (training data only).
    Random resized crop + horizontal flip + small rotation + colour jitter
    simulate new camera viewpoints / lighting, which reduces overfitting on
    a small custom dataset.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ------------------------------------------------------------- datasets ----
class ImageFolderDataset(Dataset):
    """Plain image dataset used to train / evaluate the from-scratch CNN."""

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


class FeatureDataset(Dataset):
    """Dataset over pre-extracted frozen-backbone feature vectors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------- visual helpers ----
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization so tensors can be shown as images."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
