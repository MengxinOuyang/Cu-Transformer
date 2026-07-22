"""
Training script for Cu-Transformer model.

Usage:
    python train.py

The script expects:
  - combined.csv: data file with columns Y, A-E (features), F-H (composition), X (time), I (period)
  - Image directories: B1/, B2/, S1/, S2/ matching the 'I' column values

Output:
  - best_model.pth: best model weights (by validation loss)
  - scaler.pth: feature scaler for production parameters (A-E)
"""

import os
import random
import warnings
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torchvision import transforms

from model import CuTransformer

warnings.filterwarnings("ignore")

# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Device configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================
# Dataset
# =========================
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None,
                 scaler=None, fit_scaler=False,
                 feature_cols=None, image_root='data'):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_root = image_root
        self.feature_cols = feature_cols or ['A', 'B', 'C', 'D', 'E']

        # Normalize production parameters
        raw_features = self.dataframe[self.feature_cols].values.astype(np.float32)
        if fit_scaler:
            self.scaler = StandardScaler()
            self.normalized_features = self.scaler.fit_transform(raw_features)
        elif scaler is not None:
            self.scaler = scaler
            self.normalized_features = self.scaler.transform(raw_features)
        else:
            self.scaler = None
            self.normalized_features = raw_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Load image (skip if file missing)
        img_path = os.path.join(self.image_root, str(row['I']), str(row['Y']))
        if not os.path.exists(img_path):
            return None
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            return None
        if self.transform:
            image = self.transform(image)

        # Normalized production parameters
        extra_features = torch.tensor(
            self.normalized_features[idx],
            dtype=torch.float32
        )

        # Classification label (blowing period)
        label_period = torch.tensor(int(row['I_label']), dtype=torch.long)

        # Regression label (Cu%, Fe%, S%, time-to-endpoint)
        label_regression = torch.tensor(
            row[['F', 'G', 'H', 'X']].astype(float).values,
            dtype=torch.float32
        )

        return image, extra_features, label_period, label_regression


def collate_skip_none(batch):
    """Collate function that skips None samples (missing images)."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# =========================
# Main
# =========================
if __name__ == '__main__':
    # -----------------------------------------------------------
    # Load CSV
    # -----------------------------------------------------------
    CSV_PATH = 'data/combined.csv'
    IMAGE_ROOT = 'data'
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # -----------------------------------------------------------
    # Data preprocessing
    # -----------------------------------------------------------
    numeric_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    before_drop = len(df)
    df.dropna(subset=numeric_columns, inplace=True)
    print(f"Removed {before_drop - len(df)} invalid rows")

    # Class label mapping
    class_names = sorted(df['I'].unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    df['I_label'] = df['I'].map(class_to_idx)
    num_classes = len(class_names)

    print("\nClass Mapping:")
    for k, v in class_to_idx.items():
        print(f"  {k} -> {v}")

    # -----------------------------------------------------------
    # Train / Validation / Test split (7:2:1 i.e. 70 / 20 / 10)
    # -----------------------------------------------------------
    # First split off 10% as a held-out test set
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, random_state=SEED,
        stratify=df['I_label']
    )
    # Then split train_val into 77.78%/22.22% -> 70%/20% of total
    train_df, val_df = train_test_split(
        train_val_df, test_size=2/9, random_state=SEED,
        stratify=train_val_df['I_label']
    )

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # -----------------------------------------------------------
    # Image transforms
    # -----------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # -----------------------------------------------------------
    # Create datasets (with feature normalization)
    # -----------------------------------------------------------
    feature_cols = ['A', 'B', 'C', 'D', 'E']

    train_dataset = CustomDataset(
        train_df, transform=transform,
        fit_scaler=True, feature_cols=feature_cols
    )
    # Save scaler for later use in evaluation
    scaler = train_dataset.scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to scaler.pkl")

    val_dataset = CustomDataset(
        val_df, transform=transform,
        scaler=scaler, feature_cols=feature_cols
    )
    test_dataset = CustomDataset(
        test_df, transform=transform,
        scaler=scaler, feature_cols=feature_cols
    )

    # -----------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------
    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=num_workers, collate_fn=collate_skip_none
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=num_workers, collate_fn=collate_skip_none
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=num_workers, collate_fn=collate_skip_none
    )

    # -----------------------------------------------------------
    # Model
    # -----------------------------------------------------------
    print("\nCreating model...")
    model = CuTransformer(
        num_classes=num_classes,
        num_extra_features=5
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -----------------------------------------------------------
    # Loss functions
    # -----------------------------------------------------------
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # -----------------------------------------------------------
    # Optimizer & Scheduler
    # -----------------------------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=1e-6
    )

    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------
    num_epochs = 200
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        train_cls_loss = 0.0
        train_reg_loss = 0.0

        for batch_data in train_loader:
            if batch_data is None:
                continue
            img, extra_feat, label_cls, label_reg = batch_data
            img = img.to(device)
            extra_feat = extra_feat.to(device)
            label_cls = label_cls.to(device)
            label_reg = label_reg.to(device)

            optimizer.zero_grad()

            reg_out, cls_logits = model(img, extra_feat)

            loss_cls = criterion_cls(cls_logits, label_cls)
            loss_reg = criterion_reg(reg_out, label_reg)
            total_loss = loss_cls + loss_reg

            total_loss.backward()
            optimizer.step()

            train_cls_loss += loss_cls.item()
            train_reg_loss += loss_reg.item()

        scheduler.step()

        avg_train_cls = train_cls_loss / len(train_loader)
        avg_train_reg = train_reg_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_cls_loss = 0.0
        val_reg_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data is None:
                    continue
                img, extra_feat, label_cls, label_reg = batch_data
                img = img.to(device)
                extra_feat = extra_feat.to(device)
                label_cls = label_cls.to(device)
                label_reg = label_reg.to(device)

                reg_out, cls_logits = model(img, extra_feat)

                loss_cls = criterion_cls(cls_logits, label_cls)
                loss_reg = criterion_reg(reg_out, label_reg)

                val_cls_loss += loss_cls.item()
                val_reg_loss += loss_reg.item()

                preds = torch.argmax(cls_logits, dim=1)
                val_total += label_cls.size(0)
                val_correct += (preds == label_cls).sum().item()

        avg_val_cls = val_cls_loss / len(val_loader)
        avg_val_reg = val_reg_loss / len(val_loader)
        avg_val_total = avg_val_cls + avg_val_reg
        val_accuracy = 100.0 * val_correct / val_total

        lr_current = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"LR: {lr_current:.2e} | "
            f"Train CLS: {avg_train_cls:.4f} REG: {avg_train_reg:.4f} | "
            f"Val CLS: {avg_val_cls:.4f} REG: {avg_val_reg:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # ---- Save best model ----
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'class_to_idx': class_to_idx,
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
            }, 'best_model.pth')
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    # -----------------------------------------------------------
    # Final evaluation on held-out test set
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating on held-out test set...")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_cls_loss = 0.0
    test_reg_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_data in test_loader:
            if batch_data is None:
                continue
            img, extra_feat, label_cls, label_reg = batch_data
            img = img.to(device)
            extra_feat = extra_feat.to(device)
            label_cls = label_cls.to(device)
            label_reg = label_reg.to(device)

            reg_out, cls_logits = model(img, extra_feat)

            loss_cls = criterion_cls(cls_logits, label_cls)
            loss_reg = criterion_reg(reg_out, label_reg)

            test_cls_loss += loss_cls.item()
            test_reg_loss += loss_reg.item()

            preds = torch.argmax(cls_logits, dim=1)
            test_total += label_cls.size(0)
            test_correct += (preds == label_cls).sum().item()

    avg_test_cls = test_cls_loss / len(test_loader)
    avg_test_reg = test_reg_loss / len(test_loader)
    test_accuracy = 100.0 * test_correct / test_total

    print(f"Test Classification Loss: {avg_test_cls:.4f}")
    print(f"Test Regression Loss:     {avg_test_reg:.4f}")
    print(f"Test Accuracy:            {test_accuracy:.2f}%")
    print(f"Best Validation Loss:     {best_val_loss:.4f}")

    print("\nTraining finished!")
