"""
Statistical significance tests for Cu-Transformer vs baselines.

Performs:
  - Paired t-test on per-sample regression errors (Cu, Fe, S, Time)
  - McNemar's test on endpoint prediction accuracy

Usage:
    python statistical_test.py

Requires: combined.csv, image directories, best_model.pth, scaler.pkl
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel, chi2

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

PERIOD_IMAGE_DIR = {'B1': 'B1', 'B2': 'B2', 'S1': 'S1', 'S2': 'S2'}
PERIOD_MAP = {
    'B1': ['B1', 'B1F'], 'B2': ['B2', 'B2F'],
    'S1': ['S1', 'S1F'], 'S2': ['S2', 'S2F'],
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
# Dataset
# =========================
class TestDataset(Dataset):
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
    df = df.copy()
    df['period'] = df['I']
    def make_path(row):
        d = PERIOD_IMAGE_DIR.get(row['I'], row['I'])
        return os.path.join(image_root, d, str(row['Y']))
    df['img_path'] = df.apply(make_path, axis=1)
    return df


# =========================
# Collect per-sample errors
# =========================
def collect_errors(model, loader, device):
    """Return per-sample regression errors and endpoint correctness."""
    model.eval()
    reg_errors = []       # [N, 4]: absolute errors for Cu, Fe, S, Time
    endpoint_correct = []  # [N]: 1 if correct, 0 otherwise

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, extras, targets, periods = batch
            images = images.to(device)
            extras = extras.to(device)
            targets_np = targets.cpu().numpy()

            reg_out, _ = model(images, extras)
            reg_pred = reg_out.cpu().numpy()

            for i in range(reg_pred.shape[0]):
                errors = np.abs(reg_pred[i] - targets_np[i])
                reg_errors.append(errors)

                pred_ep = check_endpoint(
                    periods[i], reg_pred[i, 0], reg_pred[i, 1], reg_pred[i, 2]
                )
                true_ep = check_endpoint(
                    periods[i], targets_np[i, 0], targets_np[i, 1], targets_np[i, 2]
                )
                endpoint_correct.append(1 if pred_ep == true_ep else 0)

    return np.array(reg_errors), np.array(endpoint_correct)


def collect_rf_errors(rf_reg, rf_clf, X_test, y_reg_test, periods_test):
    """Collect per-sample errors for Random Forest."""
    reg_pred = rf_reg.predict(X_test)
    reg_errors = np.abs(reg_pred - y_reg_test)

    endpoint_correct = []
    for i, p in enumerate(periods_test):
        pred_ep = check_endpoint(p, reg_pred[i, 0], reg_pred[i, 1], reg_pred[i, 2])
        true_ep = check_endpoint(p, y_reg_test[i, 0], y_reg_test[i, 1], y_reg_test[i, 2])
        endpoint_correct.append(1 if pred_ep == true_ep else 0)

    return reg_errors, np.array(endpoint_correct)


# =========================
# Statistical tests
# =========================
def paired_ttest(ours_errors, other_errors):
    """Paired t-test on per-sample absolute errors for each output."""
    p_values = []
    for i in range(4):
        _, p = ttest_rel(ours_errors[:, i], other_errors[:, i])
        p_values.append(p)
    return p_values


def mcnemar_test(ours_correct, other_correct):
    """McNemar's test for paired binary outcomes."""
    b = np.sum((ours_correct == 0) & (other_correct == 1))
    c = np.sum((ours_correct == 1) & (other_correct == 0))
    if b + c == 0:
        return 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    return 1 - chi2.cdf(chi2_stat, 1)


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

    # Train/test split
    train_df, test_df = train_test_split(
        df, test_size=1 - TRAIN_SPLIT, random_state=SEED,
        stratify=df['I_label']
    )

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values.astype(np.float32))

    # Build test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_data = load_all_data(test_df, IMAGE_ROOT)
    test_ds = TestDataset(test_data, transform, scaler)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=0)

    # ---- Load Cu-Transformer ----
    print("Loading Cu-Transformer...")
    model = CuTransformer(num_classes=NUM_CLASSES,
                           num_extra_features=NUM_EXTRA_FEATURES).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    ours_errors, ours_correct = collect_errors(model, test_loader, device)

    # ---- Random Forest ----
    print("Extracting features for Random Forest...")
    from torchvision import models as tv_models
    extractor = tv_models.resnet50(
        weights=tv_models.ResNet50_Weights.IMAGENET1K_V1
    )
    extractor = torch.nn.Sequential(*list(extractor.children())[:-1])
    extractor = extractor.to(device).eval()

    # Extract features from test set
    X_test_list, y_reg_test_list, periods_test = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            images, extras, targets, periods = batch
            images = images.to(device)
            feat = extractor(images).view(images.size(0), -1)
            X = torch.cat([feat, extras.to(device)], dim=1).cpu().numpy()
            X_test_list.append(X)
            y_reg_test_list.append(targets.cpu().numpy())
            periods_test.extend(periods)
    X_test = np.vstack(X_test_list)
    y_reg_test = np.vstack(y_reg_test_list)

    # Train RF on same test features? No — we need train features too.
    # Quick: extract train features
    train_data = load_all_data(train_df, IMAGE_ROOT)
    train_ds = TestDataset(train_data, transform, scaler)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    X_train_list, y_reg_train_list = [], []
    with torch.no_grad():
        for batch in train_loader:
            if batch is None: continue
            images, extras, targets, _ = batch
            images = images.to(device)
            feat = extractor(images).view(images.size(0), -1)
            X = torch.cat([feat, extras.to(device)], dim=1).cpu().numpy()
            X_train_list.append(X)
            y_reg_train_list.append(targets.cpu().numpy())
    X_train = np.vstack(X_train_list)
    y_reg_train = np.vstack(y_reg_train_list)

    # Train RF
    print("Training Random Forest...")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=SEED,
                                    n_jobs=-1)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf_reg.fit(X_train, y_reg_train)

    # Endpoint labels for RF training
    y_ep_train = []
    for i, p in enumerate(periods_test[:len(y_reg_test)]):
        pass  # We need train period labels
    # Simpler: compute EP labels from regression targets
    train_periods = list(train_data['period'])
    y_ep_train_rf = np.array([
        1 if check_endpoint(train_periods[i],
                            y_reg_train[i, 0], y_reg_train[i, 1],
                            y_reg_train[i, 2]) else 0
        for i in range(len(y_reg_train))
    ])
    rf_clf.fit(X_train, y_ep_train_rf)

    rf_errors, rf_correct = collect_rf_errors(
        rf_reg, rf_clf, X_test, y_reg_test, periods_test
    )

    # ---- Run tests ----
    print("\n" + "=" * 70)
    print("Statistical Significance Tests (Cu-Transformer vs Random Forest)")
    print("=" * 70)

    # Paired t-tests
    p_vals = paired_ttest(ours_errors, rf_errors)
    print("\nPaired t-test on per-sample absolute errors:")
    print(f"  Cu-MAE:   t-test p = {p_vals[0]:.6f}")
    print(f"  Fe-MAE:   t-test p = {p_vals[1]:.6f}")
    print(f"  S-MAE:    t-test p = {p_vals[2]:.6f}")
    print(f"  Time-MAE: t-test p = {p_vals[3]:.6f}")

    # McNemar's test
    p_mcnemar = mcnemar_test(ours_correct, rf_correct)
    print(f"\nMcNemar's test on endpoint accuracy:")
    print(f"  p = {p_mcnemar:.6f}")

    # Summary
    print("\n" + "-" * 70)
    alpha = 0.05
    sig_tests = sum(p < alpha for p in p_vals)
    print(f"Significant regression differences (p < 0.05): {sig_tests}/4")
    print(f"Significant endpoint difference (p < 0.05): "
          f"{'Yes' if p_mcnemar < alpha else 'No'}")

    print("\nAll statistical tests completed.")
