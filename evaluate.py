"""
Comprehensive evaluation script for Cu-Transformer.

Evaluates per-period regression metrics (Cu/Fe/S/Time MAE) and endpoint
prediction accuracy based on composition thresholds from the paper.

Usage:
    python evaluate.py

Configuration (edit below or use defaults):
    EXCEL_DIR: directory containing per-period CSV files (B1F.csv, etc.)
    MODEL_PATH: path to trained model checkpoint
    SCALER_PATH: path to saved StandardScaler pickle
    IMAGE_ROOT: root directory for image folders
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from model import CuTransformer

warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
EXCEL_DIR = 'data/excel'
MODEL_PATH = 'best_model.pth'
SCALER_PATH = 'scaler.pkl'
IMAGE_ROOT = 'data'
NUM_CLASSES = 4
NUM_EXTRA_FEATURES = 5
FEATURE_COLS = ['A', 'B', 'C', 'D', 'E']
BATCH_SIZE = 32

CLASS_MAPPING = {'B1': 0, 'B2': 1, 'S1': 2, 'S2': 3}
PERIOD_IMAGE_DIR = {'B1': 'B1', 'B2': 'B2', 'S1': 'S1', 'S2': 'S2'}
PERIOD_MAP = {
    'B1': ['B1', 'B1F'], 'B2': ['B2', 'B2F'],
    'S1': ['S1', 'S1F'], 'S2': ['S2', 'S2F'],
}

DIAGNOSE = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# =========================
# Endpoint criteria (paper Table 1)
# =========================
def check_endpoint(period, cu, fe, s):
    if period == 'S1':
        return fe < 1.5
    elif period == 'S2':
        return fe < 1.0
    elif period == 'B1':
        return (s < 8.0) and (cu > 90.0)
    elif period == 'B2':
        return cu > 98.5
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
class CustomDataset(Dataset):
    def __init__(self, dataframe, image_root='.', transform=None,
                 period='B1', scaler=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_root = image_root
        self.period = period
        self.period_dir = PERIOD_IMAGE_DIR.get(period, period)
        self.label_idx = CLASS_MAPPING.get(period, 0)

        # Normalize features
        raw_features = self.dataframe[FEATURE_COLS].values.astype(np.float32)
        if scaler is not None:
            self.normalized_features = scaler.transform(raw_features)
        else:
            self.normalized_features = raw_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Image path
        if 'I' in self.dataframe.columns:
            img_path = os.path.join(str(row['I']), str(row['Y']))
        else:
            img_path = os.path.join(self.image_root, self.period_dir, str(row['Y']))

        if not os.path.exists(img_path):
            return None
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            return None

        if self.transform:
            image = self.transform(image)

        extra_features = torch.tensor(
            self.normalized_features[idx], dtype=torch.float32
        )
        label_cls = torch.tensor(self.label_idx, dtype=torch.long)
        label_reg = torch.tensor(
            row[['F', 'G', 'H', 'X']].astype(float).values,
            dtype=torch.float32
        )

        return image, extra_features, label_cls, label_reg


def skip_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# =========================
# Evaluation per CSV file
# =========================
def evaluate_single_csv(csv_path, model, scaler, criterion_cls,
                        criterion_reg, period, device):
    print(f'\n{"="*70}')
    print(f'Evaluating: {os.path.basename(csv_path)} | Period: {period}')
    print('='*70)

    df = pd.read_csv(csv_path)

    # Data cleaning
    numeric_columns = FEATURE_COLS + ['F', 'G', 'H', 'X']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    before_drop = len(df)
    df.dropna(subset=numeric_columns, inplace=True)
    print(f'Data cleaning: {before_drop} -> {len(df)} '
          f'(removed {before_drop - len(df)} rows)')

    if len(df) == 0:
        print(f'[WARNING] No valid data in {csv_path}, skipping')
        return None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = CustomDataset(
        df, image_root=IMAGE_ROOT, transform=transform,
        period=period, scaler=scaler
    )
    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 1)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, collate_fn=skip_none_collate
    )

    all_cls_preds, all_cls_labels = [], []
    all_reg_preds, all_reg_labels = [], []
    running_cls_loss, running_reg_loss = 0.0, 0.0
    valid_batches = 0

    true_endpoint, pred_endpoint = [], []
    time_to_endpoint = []

    with torch.no_grad():
        for batch_data in test_loader:
            if batch_data is None:
                continue
            img, extra_feat, label_cls, label_reg = batch_data
            valid_batches += 1

            img = img.to(device)
            extra_feat = extra_feat.to(device)
            label_cls = label_cls.to(device)
            label_reg = label_reg.to(device)

            reg_out, cls_logits = model(img, extra_feat)

            cls_loss = criterion_cls(cls_logits, label_cls)
            reg_loss = criterion_reg(reg_out, label_reg)
            running_cls_loss += cls_loss.item()
            running_reg_loss += reg_loss.item()

            preds = torch.argmax(cls_logits, dim=1)
            all_cls_preds.extend(preds.cpu().numpy())
            all_cls_labels.extend(label_cls.cpu().numpy())
            all_reg_preds.extend(reg_out.cpu().numpy())
            all_reg_labels.extend(label_reg.cpu().numpy())

            # Endpoint detection
            reg_pred_np = reg_out.cpu().numpy()
            reg_true_np = label_reg.cpu().numpy()
            for i in range(reg_pred_np.shape[0]):
                cu_p, fe_p, s_p = reg_pred_np[i, 0], reg_pred_np[i, 1], reg_pred_np[i, 2]
                cu_t, fe_t, s_t = reg_true_np[i, 0], reg_true_np[i, 1], reg_true_np[i, 2]
                pred_endpoint.append(check_endpoint(period, cu_p, fe_p, s_p))
                true_endpoint.append(check_endpoint(period, cu_t, fe_t, s_t))
                time_to_endpoint.append(reg_true_np[i, 3])

    print(f'Valid batches: {valid_batches}')
    if valid_batches == 0:
        print(f'[WARNING] All batches skipped for {csv_path}')
        return None

    all_cls_preds = np.array(all_cls_preds)
    all_cls_labels = np.array(all_cls_labels)
    all_reg_preds = np.array(all_reg_preds)
    all_reg_labels = np.array(all_reg_labels)
    true_endpoint = np.array(true_endpoint)
    pred_endpoint = np.array(pred_endpoint)
    time_to_endpoint = np.array(time_to_endpoint)

    N = len(all_cls_preds)

    # Classification metrics
    cls_acc = accuracy_score(all_cls_labels, all_cls_preds)
    cls_f1 = f1_score(all_cls_labels, all_cls_preds,
                      average='weighted', zero_division=0)

    # Regression metrics
    per_mae = []
    for i in range(4):
        per_mae.append(mean_absolute_error(
            all_reg_labels[:, i], all_reg_preds[:, i]
        ))
    overall_mae = mean_absolute_error(all_reg_labels, all_reg_preds)
    overall_r2 = r2_score(all_reg_labels, all_reg_preds,
                          multioutput='uniform_average')

    # Endpoint accuracy
    ep_acc_all = accuracy_score(true_endpoint, pred_endpoint)
    near_mask = time_to_endpoint <= 30
    ep_acc_near = None
    if near_mask.sum() > 0:
        ep_acc_near = accuracy_score(
            true_endpoint[near_mask], pred_endpoint[near_mask]
        )

    # Print results
    print(f'\n--- Classification ---')
    print(f'Accuracy : {cls_acc:.4f}  F1: {cls_f1:.4f}')
    print(f'Pred distribution: '
          f'{dict(zip(*np.unique(all_cls_preds, return_counts=True)))}')

    print(f'\n--- Regression (paper Table 2) ---')
    print(f'Cu-MAE: {per_mae[0]:.4f}  Fe-MAE: {per_mae[1]:.4f}  '
          f'S-MAE: {per_mae[2]:.4f}  Time-MAE: {per_mae[3]:.4f}')
    print(f'Overall MAE: {overall_mae:.4f}  R^2: {overall_r2:.4f}')

    print(f'\n--- Endpoint Accuracy (paper Table 3) ---')
    criteria = {'S1': 'Fe < 1.5%', 'S2': 'Fe < 1%',
                'B1': 'S < 8% and Cu > 90%', 'B2': 'Cu > 98.5%'}
    print(f'Criterion: {criteria.get(period, "N/A")}')
    print(f'All samples: {ep_acc_all*100:.2f}% '
          f'({true_endpoint.sum()}/{N})')
    if ep_acc_near is not None:
        print(f'Near-endpoint (X<=30min): {ep_acc_near*100:.2f}% '
              f'({true_endpoint[near_mask].sum()}/{near_mask.sum()})')

    return {
        'file': os.path.basename(csv_path),
        'period': period, 'N': N,
        'cls_acc': cls_acc, 'cls_f1': cls_f1,
        'cu_mae': per_mae[0], 'fe_mae': per_mae[1],
        's_mae': per_mae[2], 'time_mae': per_mae[3],
        'overall_mae': overall_mae, 'overall_r2': overall_r2,
        'ep_acc_all': ep_acc_all, 'ep_acc_near': ep_acc_near,
    }


# =========================
# Main
# =========================
if __name__ == '__main__':
    # Scan CSV files
    csv_files = []
    if os.path.isdir(EXCEL_DIR):
        for f in sorted(os.listdir(EXCEL_DIR)):
            if f.lower().endswith('.csv'):
                csv_files.append(os.path.join(EXCEL_DIR, f))
        print(f'Found {len(csv_files)} CSV files in [{EXCEL_DIR}/]')
    else:
        for f in sorted(os.listdir('.')):
            if f.lower().endswith('.csv') and detect_period(f):
                csv_files.append(f)
        print(f'Found {len(csv_files)} CSV files in current directory')

    if len(csv_files) == 0:
        raise FileNotFoundError('No CSV files found.')

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')

    print('\nLoading model...')
    model = CuTransformer(
        num_classes=NUM_CLASSES, num_extra_features=NUM_EXTRA_FEATURES
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # Support both raw state_dict and full checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f'[Warning] Missing keys: {missing}')
    if unexpected:
        print(f'[Warning] Unexpected keys: {unexpected}')
    model.eval()
    print('Model loaded successfully!')

    # Load scaler
    scaler = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f'Scaler loaded from {SCALER_PATH}')
    else:
        print('[Warning] No scaler found; using raw feature values')

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # Evaluate each CSV
    results_list = []
    for csv_path in csv_files:
        period = detect_period(os.path.basename(csv_path))
        if period is None:
            print(f'\n[Skipping] Cannot detect period from: {csv_path}')
            continue
        result = evaluate_single_csv(
            csv_path, model, scaler, criterion_cls, criterion_reg,
            period, device
        )
        if result is not None:
            results_list.append(result)

    # Summary table
    if len(results_list) == 0:
        raise RuntimeError('All CSV evaluations failed.')

    print('\n\n' + '='*80)
    print('Summary (paper Table 3 format)')
    print('='*80)
    header = (f'{"Period":<8}{"N":<10}{"Cu-MAE":<10}{"Fe-MAE":<10}'
              f'{"S-MAE":<10}{"Time-MAE":<10}{"Overall MAE":<12}'
              f'{"R^2":<10}{"Endpoint Acc":<14}')
    print(header)
    print('-'*80)

    total_weighted_ep = 0.0
    total_samples = 0

    for r in results_list:
        ep_str = f"{r['ep_acc_all']*100:.2f}%"
        print(f"{r['period']:<8}{r['N']:<10}"
              f"{r['cu_mae']:<10.4f}{r['fe_mae']:<10.4f}"
              f"{r['s_mae']:<10.4f}{r['time_mae']:<10.4f}"
              f"{r['overall_mae']:<12.4f}{r['overall_r2']:<10.4f}"
              f"{ep_str:<14}")
        total_weighted_ep += r['ep_acc_all'] * r['N']
        total_samples += r['N']

    print('-'*80)
    overall_ep_acc = total_weighted_ep / total_samples if total_samples > 0 else 0
    print(f'\nOverall weighted endpoint accuracy: {overall_ep_acc*100:.2f}%')

    # Save summary
    summary_df = pd.DataFrame(results_list)
    summary_path = 'evaluation_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f'\nSummary saved to: {summary_path}')
    print('Evaluation complete!')
