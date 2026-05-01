import os, pickle
import keras
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow.keras.backend as K

app = Flask(__name__)

MODEL_PATH    = '../models/best_unet.keras'
FITSTATS_PATH = '../models/fit_stats.pkl'

# ── POS_WEIGHT must match exactly what was used in training ──
# Go to your notebook and copy the exact value printed there
# e.g. POS_WEIGHT = neg / (pos + 1e-8)
POS_WEIGHT = 2.72   # ← replace this with your actual value from the notebook

# ── Register all custom functions ──
@keras.saving.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f     = K.flatten(y_true)
    y_pred_f     = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


@keras.saving.register_keras_serializable()
def weighted_bce(y_true, y_pred, pos_weight=POS_WEIGHT):
    bce        = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weight_map = (tf.squeeze(y_true, axis=-1) * pos_weight +
                 (1 - tf.squeeze(y_true, axis=-1)))
    return K.mean(weight_map * bce)


@keras.saving.register_keras_serializable()
def combined_loss(y_true, y_pred):
    return weighted_bce(y_true, y_pred) + dice_loss(y_true, y_pred)


@keras.saving.register_keras_serializable()
def iou_metric(y_true, y_pred):
    y_pred_bin   = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union        = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


@keras.saving.register_keras_serializable()
def precision_metric(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    tp         = tf.reduce_sum(y_true * y_pred_bin)
    fp         = tf.reduce_sum((1 - y_true) * y_pred_bin)
    return (tp + 1e-6) / (tp + fp + 1e-6)


@keras.saving.register_keras_serializable()
def recall_metric(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    tp         = tf.reduce_sum(y_true * y_pred_bin)
    fn         = tf.reduce_sum(y_true * (1 - y_pred_bin))
    return (tp + 1e-6) / (tp + fn + 1e-6)


@keras.saving.register_keras_serializable()
def f1_metric(y_true, y_pred):
    p = precision_metric(y_true, y_pred)
    r = recall_metric(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-6)


# ── Load model with all custom objects ──
print("Loading U-Net model...")
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'combined_loss'   : combined_loss,
        'weighted_bce'    : weighted_bce,
        'dice_loss'       : dice_loss,
        'iou_metric'      : iou_metric,
        'precision_metric': precision_metric,
        'recall_metric'   : recall_metric,
        'f1_metric'       : f1_metric,
    }
)

with open(FITSTATS_PATH, 'rb') as f:
    fit_stats = pickle.load(f)

print("U-Net ready ✓")


def preprocess(X_raw, fit_stats):
    """
    Exact same preprocessing as training notebook.
    fit_stats keys: refl_mean, refl_std, dem_fill, dem_min, dem_max
    Returns: X_processed (N, H, W, 14)
    """
    X = X_raw.copy().astype(np.float32)
    N, H, W, _ = X.shape

    # ── 1. Spectral reflectance bands (0–6) ──
    REFL_BANDS = list(range(7))
    REFL_CLIP  = (0.0, 10000.0)

    for b in REFL_BANDS:
        X[..., b] = np.clip(X[..., b], *REFL_CLIP)

    refl_mean = fit_stats['refl_mean']
    refl_std  = fit_stats['refl_std']
    for i, b in enumerate(REFL_BANDS):
        X[..., b] = (X[..., b] - refl_mean[i]) / refl_std[i]

    # ── 2. QA band (7) → binary bad-pixel flag ──
    qa_int       = X_raw[..., 7].astype(np.int32)
    cloud        = ((qa_int >> 3) & 1).astype(np.float32)
    cloud_shadow = ((qa_int >> 4) & 1).astype(np.float32)
    adj_cloud    = ((qa_int >> 5) & 1).astype(np.float32)
    bad_pixel    = np.clip(cloud + cloud_shadow + adj_cloud, 0, 1)
    X[..., 7]    = bad_pixel

    # ── 3. DEM bands (8–9) → fix NoData, normalize ──
    for b in [8, 9]:
        nodata_mask = X[..., b] < -9000
        fill_val    = fit_stats['dem_fill'][b]
        b_min       = fit_stats['dem_min'][b]
        b_max       = fit_stats['dem_max'][b]

        X[..., b][nodata_mask] = fill_val
        X[..., b] = (X[..., b] - b_min) / (b_max - b_min + 1e-8)
        X[..., b] = np.clip(X[..., b], 0, 1)

    # ── 4. ESA World Cover (10) → binary water class ──
    X[..., 10] = (X_raw[..., 10].astype(int) == 80).astype(np.float32)

    # ── 5. Water occurrence probability (11) → [0, 1] ──
    X[..., 11] = np.clip(X_raw[..., 11] / 100.0, 0, 1)

    # ── 6. NDWI + MNDWI from raw reflectance ──
    raw_green = np.clip(X_raw[..., 2], 0, 10000).astype(np.float32)
    raw_nir   = np.clip(X_raw[..., 4], 0, 10000).astype(np.float32)
    raw_swir1 = np.clip(X_raw[..., 5], 0, 10000).astype(np.float32)

    ndwi  = np.clip((raw_green - raw_nir)  / (raw_green + raw_nir  + 1e-8), -1, 1)
    mndwi = np.clip((raw_green - raw_swir1)/ (raw_green + raw_swir1+ 1e-8), -1, 1)

    X = np.concatenate([
        X,
        ndwi [..., np.newaxis],
        mndwi[..., np.newaxis]
    ], axis=-1)   # (N, H, W, 14)

    return X


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data  = request.get_json()
        X_raw = np.array(data['image'], dtype=np.float32)

        if X_raw.ndim == 3:
            X_raw = X_raw[np.newaxis, ...]

        X_proc = preprocess(X_raw, fit_stats)   # ← returns X only, not (X, fit_stats)
        probs  = model.predict(X_proc, verbose=0)

        return jsonify({'probs': probs.tolist(), 'shape': list(probs.shape)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/debug', methods=['GET'])
def debug():
    return jsonify({
        'fit_stats_keys': list(fit_stats.keys()),
        'fit_stats_sample': {
            k: {'mean': str(v['mean']), 'std': str(v['std'])}
            if isinstance(v, dict) else str(v)
            for k, v in fit_stats.items()
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'unet'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)