import os, pickle
import numpy as np
from flask import Flask, request, jsonify
import torch
import segmentation_models_pytorch as smp

app = Flask(__name__)

MODEL_PATH    = '../models/best_deeplab.pth'
FITSTATS_PATH = '../models/fit_stats_deeplab.pkl'
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading DeepLab model on {DEVICE}...")

print("SMP version:", smp.__version__)
# ── Rebuild the exact same architecture as training ──
model = smp.DeepLabV3Plus(
    encoder_name    = 'efficientnet-b3',
    encoder_weights = None,          # no pretrained weights — loading from checkpoint
    in_channels     = 15,
    classes         = 1,
)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

with open(FITSTATS_PATH, 'rb') as f:
    fit_stats = pickle.load(f)

print("DeepLab ready ✓")


def preprocess(X_raw, fit_stats):
    """
    Exact same preprocessing as DeepLab training notebook.
    Returns: X_processed (N, H, W, 15)
    """
    X = X_raw.copy().astype(np.float32)

    # ── 1. Spectral reflectance (0–6) ──
    REFL_BANDS = list(range(7))
    for b in REFL_BANDS:
        X[..., b] = np.clip(X[..., b], 0.0, 10000.0)

    refl_mean = fit_stats['refl_mean']
    refl_std  = fit_stats['refl_std']
    for i, b in enumerate(REFL_BANDS):
        X[..., b] = (X[..., b] - refl_mean[i]) / refl_std[i]

    # ── 2. QA band (7) → binary bad-pixel flag ──
    qa_int = X_raw[..., 7].astype(np.int32)
    bad    = np.clip(((qa_int>>3)&1) + ((qa_int>>4)&1) + ((qa_int>>5)&1), 0, 1)
    X[..., 7] = bad.astype(np.float32)

    # ── 3. DEM (8–9) → NoData fill + min-max ──
    for b in [8, 9]:
        nd       = X[..., b] < -9000
        fill_val = fit_stats['dem_fill'][b]
        bmin     = fit_stats['dem_min'][b]
        bmax     = fit_stats['dem_max'][b]

        X[..., b][nd] = fill_val
        X[..., b] = np.clip((X[..., b] - bmin) / (bmax - bmin + 1e-8), 0, 1)

    # ── 4. ESA World Cover (10) → binary water flag ──
    X[..., 10] = (X_raw[..., 10].astype(int) == 80).astype(np.float32)

    # ── 5. Water occurrence probability (11) → [0, 1] ──
    X[..., 11] = np.clip(X_raw[..., 11] / 100.0, 0, 1)

    # ── 6. Engineered water indices ──
    g  = np.clip(X_raw[..., 2], 0, 10000).astype(np.float32)
    n  = np.clip(X_raw[..., 4], 0, 10000).astype(np.float32)
    s1 = np.clip(X_raw[..., 5], 0, 10000).astype(np.float32)
    s2 = np.clip(X_raw[..., 6], 0, 10000).astype(np.float32)

    ndwi  = np.clip((g - n)  / (g + n  + 1e-8), -1, 1)
    mndwi = np.clip((g - s1) / (g + s1 + 1e-8), -1, 1)
    awei  = 4*(g - s1) - (0.25*n + 2.75*s2)

    awei_mean = fit_stats['awei_mean']
    awei_std  = fit_stats['awei_std']
    awei_norm = (awei - awei_mean) / awei_std

    X = np.concatenate([
        X,
        ndwi     [..., np.newaxis],
        mndwi    [..., np.newaxis],
        awei_norm[..., np.newaxis]
    ], axis=-1)   # (N, H, W, 15)

    return X


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data  = request.get_json()
        X_raw = np.array(data['image'], dtype=np.float32)

        if X_raw.ndim == 3:
            X_raw = X_raw[np.newaxis, ...]

        X_proc = preprocess(X_raw, fit_stats)

        # PyTorch expects (N, C, H, W)
        tensor = torch.tensor(X_proc).permute(0, 3, 1, 2).float().to(DEVICE)

        with torch.no_grad():
            probs = torch.sigmoid(model(tensor)).cpu().numpy()  # (N,1,H,W)

        # Convert to (N, H, W, 1) to match U-Net format
        probs = probs.transpose(0, 2, 3, 1)

        return jsonify({'probs': probs.tolist(), 'shape': list(probs.shape)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'deeplab'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)