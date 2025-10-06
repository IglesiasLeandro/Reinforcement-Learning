import os
import glob
from typing import Counter
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------------
# Dataset BUSI
# -----------------------------
class BUSIDataset(Dataset):
    def __init__(self, files, size=256, augment=False):
        """
        files: lista de caminhos das imagens originais (sem _mask)
        size: redimensionar para (size, size)
        augment: aplicar augmentations (Albumentations)
        """
        self.files = files
        self.size = size
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ElasticTransform(alpha=50, sigma=7, p=0.3),   # removido alpha_affine
                A.GaussNoise(mean=0, std=10, p=0.3),            # usa std no lugar de var_limit
                A.Resize(size, size),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(size, size),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        mask_path = img_path.replace(".png", "_mask.png")

        # grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # garante np.float32
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # Albumentations espera HWC
        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]        # [1,H,W]
        mask_t = augmented["mask"].unsqueeze(0).float() if augmented["mask"].ndim == 2 else augmented["mask"].float()

        return img_t, mask_t

# -----------------------------
# Funções auxiliares
# -----------------------------
def load_busi_paths(root_dir):
    """Lista todos os caminhos de imagens (sem máscaras)."""
    all_imgs = []
    classes = ["benign", "malignant", "normal"]
    for c in classes:
        folder = os.path.join(root_dir, c)
        for f in glob.glob(os.path.join(folder, "*.png")):
            if "_mask" not in f:
                all_imgs.append((f, c))
    return all_imgs

def split_busi(root_dir, test_size=0.1, val_size=0.1, seed=42):
    all_imgs = load_busi_paths(root_dir)  # [(path, classe), ...]
    paths = [p for p,c in all_imgs]
    labels = [c for p,c in all_imgs]

    # primeiro split train+val vs test
    trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
        paths, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )

    # agora split train vs val (usando trainval_labels para estratificar)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        trainval_paths, trainval_labels,
        test_size=val_size,
        random_state=seed,
        stratify=trainval_labels
    )
    print("Treino", Counter(train_labels))
    print("Validação", Counter(val_labels))
    print("Teste", Counter(test_labels))
    return train_paths, val_paths, test_paths

# -----------------------------
# DataLoaders prontos
# -----------------------------
def get_busi_loaders(root_dir, size=256, batch_size=8, num_workers=2):
    train_files, val_files, test_files = split_busi(root_dir)

    train_ds = BUSIDataset(train_files, size=size, augment=True)
    val_ds   = BUSIDataset(val_files, size=size, augment=False)
    test_ds  = BUSIDataset(test_files, size=size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader