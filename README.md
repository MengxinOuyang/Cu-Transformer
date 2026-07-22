# Cu-Transformer: Integrated Prediction Model for Copper Converter Blowing

Official implementation of **"Transformer-Based Integrated Prediction Model for Copper Converter Blowing Endpoint and Composition"**.

The Cu-Transformer model integrates **RepViT** (CVPR 2024), **SHViT** (single-head vision transformer), and **LiteMLA** (ICCV 2023) modules within a Swin-style hierarchical framework for simultaneous prediction of blowing endpoints and melt composition in copper converter operations.

## Environment Setup

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- timm (for SqueezeExcite layer used in RepViT and SHViT)
- pandas, scikit-learn, numpy, Pillow
- matplotlib, tqdm, scipy (for comparison experiments and statistical tests)

### Installation

```bash
pip install torch torchvision timm pandas scikit-learn numpy Pillow matplotlib tqdm scipy
```

## File Structure

```
cu-transformer/
├── model.py                  # Cu-Transformer architecture (class CuTransformer)
├── repvit.py                 # RepViT block (CVPR 2024) with structural reparameterization
├── SHViTBlock.py             # Single-Head Vision Transformer block
├── SwinTransformer.py        # Swin Transformer components (PatchEmbed, PatchMerging)
├── LiteMLA.py                # Lightweight Multi-Scale Linear Attention (ICCV 2023)
├── train.py                  # Training script with normalization, scheduler, held-out test
├── evaluate.py               # Per-period evaluation with endpoint accuracy metrics
├── compare.py                # Benchmark comparison (Random Forest, MLP, ResNet-50)
├── statistical_test.py       # Statistical significance tests (paired t-test + McNemar)
└── README.md
```

## Data Format

The model expects a CSV file (`combined.csv`) with the following columns:

| Column | Description |
|--------|-------------|
| `Y` | Image filename (e.g., `25 (1).jpg`) |
| `A` | Production parameter 1 |
| `B` | Production parameter 2 |
| `C` | Production parameter 3 |
| `D` | Production parameter 4 |
| `E` | Production parameter 5 |
| `F` | Cu composition (%) |
| `G` | Fe composition (%) |
| `H` | S composition (%) |
| `X` | Time to endpoint (minutes) |
| `I` | Blowing period label (`B1`, `B2`, `S1`, `S2`) |

Images should be organized in subdirectories named by period:
```

B1/   # Slagging stage 1 images
B2/   # Slagging stage 2 images
S1/   # Copper-forming stage 1 images
S2/   # Copper-forming stage 2 images
```

For per-period evaluation, place per-period CSV files in an `excel/` directory:
```
excel/B1F.csv, excel/B2F.csv, excel/S1F.csv, excel/S2F.csv
```

## Training

```bash
python train.py
```

The training script:
- Loads data from `combined.csv` and normalizes production parameters (StandardScaler)
- Splits into **train (70%) / validation (20%) / held-out test (10%)** via pure random split without stratified sampling
- Trains for up to 200 epochs with:
- SGD optimizer (lr=0.01, momentum=0.9)
-CosineAnnealingLR scheduler (eta_min=1e-6)
-Early stopping (patience=30 epochs)
-Combined loss: CrossEntropyLoss (period classification) + MSELoss (regression for Cu/Fe/S/Time)
- Saves:
  - `best_model.pth` — best model checkpoint (includes weights, scaler, class mapping)
  - `scaler.pkl` — feature normalization scaler

Key hyperparameters (modifiable in `train.py`):
- `num_classes=4` (blowing periods: B1, B2, S1, S2)
- `num_extra_features=5` (production parameters A-E)
- `batch_size=32`, image size 224×224
- `embed_dim=96` (Swin-Tiny base)

## Evaluation

### Per-Period Evaluation with Endpoint Metrics

```bash
python evaluate.py
```

Provides per-period results:
- Regression metrics: Cu-MAE, Fe-MAE, S-MAE, Time-MAE, overall R²
- Endpoint prediction accuracy based on composition thresholds (paper Table 1):
  - **S1** (copper-forming 1): Fe < 1.5%
  - **S2** (copper-forming 2): Fe < 1.0%
  - **B1** (slagging 1): S < 8% and Cu > 90%
  - **B2** (slagging 2): Cu > 98.5%
- Summary table saved to `evaluation_summary.csv`

### Benchmark Comparison

```bash
python compare.py
```

Compares Cu-Transformer against:
- **Random Forest** (ResNet-50 features + production parameters)
- **MLP** (ResNet-18 features + production parameters)
- **ResNet-50 + MLP**

Reports per-output MAE, R², endpoint accuracy, McNemar p-value, and parameter count.
Results saved to `comparison_results.csv`.

### Statistical Significance Tests

```bash
python statistical_test.py
```

Performs:
- **Paired t-tests** on per-sample absolute regression errors (Cu, Fe, S, Time)
- **McNemar's test** on endpoint prediction accuracy

Between Cu-Transformer and Random Forest baselines.

## Model Architecture

```
Input: 3×224×224 image + 5 production parameters
  │
  ├── Channel concatenation (8 channels)
  ├── PatchEmbed (4×4 patches → 96-dim, 56×56 spatial)
  │
  ├── Stage 1: RepViTBlock ×2 ── PatchMerging → 192-dim, 28×28
  ├── Stage 2: RepViTBlock ×2 ── PatchMerging → 384-dim, 14×14
  ├── Stage 3: RepViTBlock ×6 ──┐
  │                              ├── Auxiliary: RepViTBlock → GAP → LN → Period classification (4-class)
  │                              │
  ├── PatchMerging → 768-dim, 7×7
  │
  ├── Stage 4: SHViTBlock (single-head self-attention)
  ├── LiteMLA (multi-scale linear attention, kernel=5)
  ├── Global Average Pooling
  ├── LayerNorm
  └── Regression Head → [Cu%, Fe%, S%, Time-to-endpoint]
       │
       └── Composition constraint: Cu% + Fe% + S% = 100%
```

### Key Components

- **RepViTBlock** (CVPR 2024): RepVGG-style depthwise separable convolutions with structural reparameterization — multi-branch during training, fused single-path during inference
- **SHViTBlock**: Single-head self-attention applied to a subset of channels, reducing computational redundancy vs. multi-head attention
- **LiteMLA** (ICCV 2023): ReLU-based linear attention with multi-scale aggregation kernels, offering O(N) complexity vs. O(N²) softmax attention
- **Swin macro-architecture**: Hierarchical design with PatchEmbed/PatchMerging for efficient multi-scale feature extraction

## Reproducibility

All scripts set random seeds (`SEED=42`) for Python `random`, NumPy, and PyTorch.
CuDNN deterministic mode is enabled. Training uses stratified splits with fixed `random_state`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ouyang2026transformer,
  title={Transformer-Based Integrated Prediction Model for Copper Converter Blowing Endpoint and Composition},
  author={Ouyang, Mengxin and Qiu, Yunhao and Li, Mingzhou and Wan, Zhanghao and Huang, Jindi and Zhang, Fuquan and Yuan, Shixiong and Zhong, Lihua},
  journal={TBD},
  year={2026}
}
```

## License

This project is available for academic and research purposes.
