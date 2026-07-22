"""
Benchmark comparison: Cu-Transformer vs Random Forest, MLP, ResNet-50.

All models are evaluated on the SAME train/test split for fair comparison.
Statistical significance assessed via McNemar's test on endpoint accuracy.

Usage:
    python compare.py

Requires: combined.csv, image directories, and the trained best_model.pth + scaler.pkl
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import CuTransformer

warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
CSV_PATH = 'data/combined.csv'
MODEL_PATH = 'best_model.pth'
SCALER_PATH = 'scaler.pkl'
IMAGE_ROOT = 'data'
FEATURE_COLS = ['A', 'B', 'C', 'D', 'E']
NUM_CLASSES = 4
NUM_EXTRA_FEATURES = 5
BATCH_SIZE = 32
SEED = 42
TRAIN_SPLIT = 0.9
EPOCHS = 30
LR = 1e-4

CACHE_DIR = 'model_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

PERIOD_IMAGE_DIR = {'B1': 'B1', 'B2': 'B2', 'S1': 'S1', 'S2': 'S2'}
PERIOD_MAP = {
    'B1': ['B1', 'B1F'], 'B2': ['B2', 'B2F'],
    'S1': ['S1', 'S1F'], 'S2': ['S2', 'S2F'],
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# Utility functions
# =========================
def check_endpoint(period, cu, fe, s):
    if period == 'S1': return fe < 1.5
    elif period == 'S2': return fe < 1.0
    elif period == 'B1': return (s < 8.0) and (cu > 90.0)
    elif period == 'B2': return cu > 98.5
    return False


def detect_period(filename):
    base = os.path.splitext(filename)[0].upper()
    for period, keywords in PERIOD_MAP.items():
        for kw in keywords:
            if kw.upper() in base:
                return period
    return None


# =========================
# Data loading
# =========================
class UnifiedDataset(Dataset):
    def __init__(self, dataframe, transform=None, scaler=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        raw_features = self.dataframe[FEATURE_COLS].values.astype(np.float32)
        if scaler is not None:
            self.normalized_features = scaler.transform(raw_features)
        else:
            self.normalized_features = raw_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['img_path']
        if not os.path.exists(img_path):
            return None
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            return None
        if self.transform:
            image = self.transform(image)

        extra = torch.tensor(self.normalized_features[idx], dtype=torch.float32)
        target = torch.tensor(
            row[['F', 'G', 'H', 'X']].astype(float).values,
            dtype=torch.float32
        )
        period = row['period']
        return image, extra, target, period


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images, extras, targets, periods = zip(*batch)
    return (torch.stack(images), torch.stack(extras),
            torch.stack(targets), periods)


def load_all_data(df, image_root):
    """Build UnifiedDataset-compatible dataframe from loaded CSV."""
    df = df.copy()
    df['period'] = df['I']
    # Build image paths
    def make_path(row):
        period_dir = PERIOD_IMAGE_DIR.get(row['I'], row['I'])
        return os.path.join(image_root, period_dir, str(row['Y']))
    df['img_path'] = df.apply(make_path, axis=1)
    return df


# =========================
# Cache helpers
# =========================
def get_cache_path(name, suffix):
    return os.path.join(CACHE_DIR, f"{name}_{suffix}")


def load_metrics_cache(name):
    path = get_cache_path(name, "metrics.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_metrics_cache(name, metrics):
    metrics_copy = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_copy[k] = v.tolist()
        else:
            metrics_copy[k] = v
    with open(get_cache_path(name, "metrics.json"), 'w') as f:
        json.dump(metrics_copy, f, indent=2)


# =========================
# Feature extractor for RF
# =========================
def build_feature_extractor(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device).eval()
    return model, 2048


def extract_features(dataset, extractor, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_fn, shuffle=False)
    X_list, y_reg_list, y_ep_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            if batch is None:
                continue
            images, extras, targets, periods = batch
            images = images.to(device)
            feat_img = extractor(images).view(images.size(0), -1)
            X = torch.cat([feat_img, extras.to(device)], dim=1).cpu().numpy()
            X_list.append(X)
            y_reg_list.append(targets.cpu().numpy())
            for i, p in enumerate(periods):
                cu, fe, s = targets[i, 0].item(), targets[i, 1].item(), targets[i, 2].item()
                y_ep_list.append(1 if check_endpoint(p, cu, fe, s) else 0)
    return (np.vstack(X_list), np.vstack(y_reg_list), np.array(y_ep_list))


# =========================
# McNemar test
# =========================
def mcnemar_pvalue(ours_correct, other_correct):
    ours_correct = np.array(ours_correct).astype(int)
    other_correct = np.array(other_correct).astype(int)
    b = np.sum((ours_correct == 0) & (other_correct == 1))
    c = np.sum((ours_correct == 1) & (other_correct == 0))
    if b + c == 0:
        return 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    return 1 - chi2.cdf(chi2_stat, 1)


# =========================
# Baseline models
# =========================
class RandomForestModel:
    def __init__(self):
        self.reg = RandomForestRegressor(
            n_estimators=100, random_state=SEED, n_jobs=-1
        )
        self.clf = RandomForestClassifier(
            n_estimators=100, random_state=SEED
        )

    def fit(self, X, y_reg, y_ep):
        self.reg.fit(X, y_reg)
        self.clf.fit(X, y_ep)

    def predict(self, X):
        return self.reg.predict(X), self.clf.predict(X)


class MLPModel(nn.Module):
    def __init__(self, img_encoder, extra_dim, hidden=256):
        super().__init__()
        self.img_encoder = img_encoder
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            img_dim = self.img_encoder(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(img_dim + extra_dim, hidden),
            nn.ReLU(), nn.Linear(hidden, 4)
        )

    def forward(self, img, extra):
        return self.fc(torch.cat([self.img_encoder(img), extra], dim=1))


class ResNetMLP(nn.Module):
    def __init__(self, img_model, extra_dim, hidden=256):
        super().__init__()
        self.img_encoder = img_model
        self.img_encoder.fc = nn.Identity()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            img_dim = self.img_encoder(dummy).shape[1]
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + extra_dim, hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Linear(hidden // 2, 4)
        )

    def forward(self, img, extra):
        return self.fusion(torch.cat([self.img_encoder(img), extra], dim=1))


def train_baseline(model, train_loader, val_loader, epochs, device, name):
    model.to(device)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if batch is None: continue
            img, extra, targets, _ = batch
            img, extra, targets = img.to(device), extra.to(device), targets.to(device)
            opt.zero_grad()
            loss = criterion(model(img, extra), targets)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                img, extra, targets, _ = batch
                img, extra, targets = img.to(device), extra.to(device), targets.to(device)
                val_loss += criterion(model(img, extra), targets).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),
                       get_cache_path(name, "best.pth"))

    # Load best
    best_path = get_cache_path(name, "best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    return model


def evaluate_baseline_regression(model, loader, device):
    model.eval()
    all_pred, all_true, all_periods = [], [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            img, extra, targets, periods = batch
            img, extra = img.to(device), extra.to(device)
            pred = model(img, extra).cpu().numpy()
            all_pred.append(pred)
            all_true.append(targets.cpu().numpy())
            all_periods.extend(periods)
    reg_pred = np.vstack(all_pred)
    reg_true = np.vstack(all_true)
    ep_pred, ep_true = [], []
    for i, p in enumerate(all_periods):
        ep_pred.append(1 if check_endpoint(p, reg_pred[i, 0], reg_pred[i, 1], reg_pred[i, 2]) else 0)
        ep_true.append(1 if check_endpoint(p, reg_true[i, 0], reg_true[i, 1], reg_true[i, 2]) else 0)
    return reg_pred, reg_true, np.array(ep_pred), np.array(ep_true)


# =========================
# Main
# =========================
if __name__ == '__main__':
    # ---- Load data ----
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    numeric_cols = FEATURE_COLS + ['F', 'G', 'H', 'X']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)
    class_names = sorted(df['I'].unique())
    df['I_label'] = df['I'].map({n: i for i, n in enumerate(class_names)})

    # Train/test split (same seed as training)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=1 - TRAIN_SPLIT,
        random_state=SEED, stratify=df['I_label']
    )
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # ---- Fit scaler on training data ----
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values.astype(np.float32))

    # ---- Build datasets ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = load_all_data(train_df, IMAGE_ROOT)
    test_data = load_all_data(test_df, IMAGE_ROOT)

    train_ds = UnifiedDataset(train_data, transform, scaler)
    test_ds = UnifiedDataset(test_data, transform, scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ---- 1. Evaluate Cu-Transformer ----
    print("\nEvaluating Cu-Transformer...")
    ours_metrics = load_metrics_cache("CuTransformer")

    if ours_metrics is None:
        model = CuTransformer(num_classes=NUM_CLASSES,
                               num_extra_features=NUM_EXTRA_FEATURES).to(device)
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        all_reg_pred, all_reg_true = [], []
        all_ep_pred, all_ep_true = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Cu-Transformer eval"):
                if batch is None: continue
                img, extra, targets, periods = batch
                img, extra = img.to(device), extra.to(device)
                reg_out, _ = model(img, extra)
                reg_pred = reg_out.cpu().numpy()
                reg_true = targets.cpu().numpy()
                all_reg_pred.append(reg_pred)
                all_reg_true.append(reg_true)
                for i, p in enumerate(periods):
                    all_ep_pred.append(1 if check_endpoint(
                        p, reg_pred[i, 0], reg_pred[i, 1], reg_pred[i, 2]) else 0)
                    all_ep_true.append(1 if check_endpoint(
                        p, reg_true[i, 0], reg_true[i, 1], reg_true[i, 2]) else 0)

        reg_pred = np.vstack(all_reg_pred)
        reg_true = np.vstack(all_reg_true)
        ep_pred = np.array(all_ep_pred)
        ep_true = np.array(all_ep_true)

        ours_metrics = {
            'cu_mae': mean_absolute_error(reg_true[:, 0], reg_pred[:, 0]),
            'fe_mae': mean_absolute_error(reg_true[:, 1], reg_pred[:, 1]),
            's_mae': mean_absolute_error(reg_true[:, 2], reg_pred[:, 2]),
            'time_mae': mean_absolute_error(reg_true[:, 3], reg_pred[:, 3]),
            'overall_r2': r2_score(reg_true, reg_pred,
                                    multioutput='uniform_average'),
            'overall_acc': accuracy_score(ep_true, ep_pred),
            'endpoint_correct': (ep_pred == ep_true).astype(int).tolist(),
        }
        save_metrics_cache("CuTransformer", ours_metrics)

    ours_params = sum(p.numel() for p in CuTransformer(
        num_classes=NUM_CLASSES, num_extra_features=NUM_EXTRA_FEATURES
    ).parameters()) / 1e6

    results = {
        "Cu-Transformer (Ours)": {
            "metrics": ours_metrics, "params": ours_params
        }
    }
    ours_ep_correct = np.array(ours_metrics['endpoint_correct'])

    # ---- 2. Random Forest ----
    print("\nTraining Random Forest...")
    extractor, _ = build_feature_extractor(device)
    X_train, y_reg_train, y_ep_train = extract_features(
        train_ds, extractor, device)
    X_test, y_reg_test, y_ep_test = extract_features(
        test_ds, extractor, device)

    rf = RandomForestModel()
    rf.fit(X_train, y_reg_train, y_ep_train)
    rf_reg_pred, rf_ep_pred = rf.predict(X_test)
    rf_ep_correct = (rf_ep_pred == y_ep_test).astype(int)

    rf_metrics = {
        'cu_mae': mean_absolute_error(y_reg_test[:, 0], rf_reg_pred[:, 0]),
        'fe_mae': mean_absolute_error(y_reg_test[:, 1], rf_reg_pred[:, 1]),
        's_mae': mean_absolute_error(y_reg_test[:, 2], rf_reg_pred[:, 2]),
        'time_mae': mean_absolute_error(y_reg_test[:, 3], rf_reg_pred[:, 3]),
        'overall_r2': r2_score(y_reg_test, rf_reg_pred,
                                multioutput='uniform_average'),
        'overall_acc': accuracy_score(y_ep_test, rf_ep_pred),
        'endpoint_correct': rf_ep_correct.tolist(),
    }
    results["Random Forest"] = {
        "metrics": rf_metrics, "params": 0.5  # approximate
    }

    # ---- 3. MLP with ResNet-18 features ----
    print("\nTraining MLP (ResNet-18)...")
    mlp = MLPModel(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
                   NUM_EXTRA_FEATURES)
    mlp = train_baseline(mlp, train_loader, test_loader, EPOCHS, device, "MLP")
    _, _, mlp_ep_pred, mlp_ep_true = evaluate_baseline_regression(
        mlp, test_loader, device)
    mlp_ep_correct = (mlp_ep_pred == mlp_ep_true).astype(int)

    mlp_reg_pred, mlp_reg_true, _, _ = evaluate_baseline_regression(
        mlp, test_loader, device)
    mlp_metrics = {
        'cu_mae': mean_absolute_error(mlp_reg_true[:, 0], mlp_reg_pred[:, 0]),
        'fe_mae': mean_absolute_error(mlp_reg_true[:, 1], mlp_reg_pred[:, 1]),
        's_mae': mean_absolute_error(mlp_reg_true[:, 2], mlp_reg_pred[:, 2]),
        'time_mae': mean_absolute_error(mlp_reg_true[:, 3], mlp_reg_pred[:, 3]),
        'overall_r2': r2_score(mlp_reg_true, mlp_reg_pred,
                                multioutput='uniform_average'),
        'overall_acc': accuracy_score(mlp_ep_true, mlp_ep_pred),
        'endpoint_correct': mlp_ep_correct.tolist(),
    }
    mlp_params = sum(p.numel() for p in mlp.parameters()) / 1e6
    results["MLP (ResNet-18)"] = {"metrics": mlp_metrics, "params": mlp_params}

    # ---- 4. ResNet-50 + MLP ----
    print("\nTraining ResNet-50 + MLP...")
    rn50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    rn50.fc = nn.Identity()
    resnet_model = ResNetMLP(rn50, NUM_EXTRA_FEATURES)
    resnet_model = train_baseline(resnet_model, train_loader, test_loader,
                                   EPOCHS, device, "ResNet50")
    rn_reg_pred, rn_reg_true, rn_ep_pred, rn_ep_true = \
        evaluate_baseline_regression(resnet_model, test_loader, device)
    rn_ep_correct = (rn_ep_pred == rn_ep_true).astype(int)

    rn_metrics = {
        'cu_mae': mean_absolute_error(rn_reg_true[:, 0], rn_reg_pred[:, 0]),
        'fe_mae': mean_absolute_error(rn_reg_true[:, 1], rn_reg_pred[:, 1]),
        's_mae': mean_absolute_error(rn_reg_true[:, 2], rn_reg_pred[:, 2]),
        'time_mae': mean_absolute_error(rn_reg_true[:, 3], rn_reg_pred[:, 3]),
        'overall_r2': r2_score(rn_reg_true, rn_reg_pred,
                                multioutput='uniform_average'),
        'overall_acc': accuracy_score(rn_ep_true, rn_ep_pred),
        'endpoint_correct': rn_ep_correct.tolist(),
    }
    rn_params = sum(p.numel() for p in resnet_model.parameters()) / 1e6
    results["ResNet-50 + MLP"] = {
        "metrics": rn_metrics, "params": rn_params
    }

    # ---- 5. Build comparison table ----
    print("\n" + "=" * 90)
    print("Comparison Results")
    print("=" * 90)

    table_data = []
    for name, res in results.items():
        m = res["metrics"]
        # Compute McNemar p-value vs Cu-Transformer
        if name == "Cu-Transformer (Ours)":
            p_val = "-"
        else:
            other_ep = np.array(m['endpoint_correct'])
            p_val = mcnemar_pvalue(ours_ep_correct, other_ep)
            p_val = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"

        table_data.append({
            "Model": name,
            "Cu-MAE": f"{m['cu_mae']:.2f}",
            "Fe-MAE": f"{m['fe_mae']:.2f}",
            "S-MAE": f"{m['s_mae']:.2f}",
            "Time-MAE": f"{m['time_mae']:.2f}",
            "R^2": f"{m['overall_r2']:.4f}",
            "Endpoint Acc": f"{m['overall_acc']*100:.2f}%",
            "p-value": p_val,
            "Params (M)": f"{res['params']:.1f}",
        })

    df_table = pd.DataFrame(table_data)
    print(df_table.to_string(index=False))

    df_table.to_csv("comparison_results.csv", index=False,
                    encoding='utf-8-sig')
    print("\nResults saved to comparison_results.csv")
