import io, base64, os, requests
import numpy as np
from flask import Flask, request, render_template, jsonify
import rasterio
import concurrent.futures

app = Flask(__name__)

UNET_URL       = os.getenv('UNET_URL',       'http://localhost:5001/predict')
DEEPLAB_URL    = os.getenv('DEEPLAB_URL',     'http://localhost:5002/predict')
UNET_HEALTH    = os.getenv('UNET_HEALTH',    'http://localhost:5001/health')
DEEPLAB_HEALTH = os.getenv('DEEPLAB_HEALTH', 'http://localhost:5002/health')
BEST_THRESHOLD = 0.5


def read_tif(file_bytes, is_mask=False):
    """Read a .tif file and return numpy array."""
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        arr = src.read().astype(np.float32)   # (C, H, W)
        arr = arr.transpose(1, 2, 0)          # (H, W, C)

    if is_mask:
        # Take only first band regardless of how many channels the mask has
        arr = arr[:, :, 0]                    # (H, W)
        # Binarize — handle both 0/1 masks and 0/255 masks
        arr = (arr > 0).astype(np.uint8)

    return arr


# def mask_to_png_base64(mask_2d):
#     """Convert a 2D binary mask to a base64 PNG."""
#     # Scale 0/1 → 0/255 for clear black/white rendering
#     img_array = (mask_2d * 255).astype(np.uint8)

#     # Water = cyan, background = dark — looks great on the dark UI
#     h, w = img_array.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)

#     # Background → dark navy
#     rgb[img_array == 0] = [7, 13, 26]

#     # Water pixels → cyan
#     rgb[img_array > 0]  = [6, 182, 212]

#     from PIL import Image
#     img = Image.fromarray(rgb, mode='RGB')
#     buf = io.BytesIO()
#     img.save(buf, format='PNG')
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode('utf-8')

def mask_to_png_base64(X_raw_single, mask_2d):
    """
    Overlay the predicted mask on the RGB composite of the satellite image.
    X_raw_single: (128, 128, 12) — raw bands
    mask_2d     : (128, 128)    — binary mask
    """
    # ── Build RGB composite from bands Red(3), Green(2), Blue(1) ──
    red   = X_raw_single[:, :, 3].astype(np.float32)
    green = X_raw_single[:, :, 2].astype(np.float32)
    blue  = X_raw_single[:, :, 1].astype(np.float32)

    # Clip to valid reflectance range and normalize to [0, 255]
    def norm_band(b):
        b = np.clip(b, 0, 3000)   # typical HLS surface reflectance range
        b = (b / 3000 * 255).astype(np.uint8)
        return b

    rgb = np.stack([norm_band(red),
                    norm_band(green),
                    norm_band(blue)], axis=-1)   # (128, 128, 3)

    # ── Overlay water mask as cyan with transparency ──
    overlay = rgb.copy()
    water_pixels = mask_2d > 0

    # Blend water pixels: 40% original + 60% cyan
    cyan = np.array([6, 182, 212], dtype=np.float32)
    overlay[water_pixels] = (
        0.35 * rgb[water_pixels].astype(np.float32) +
        0.65 * cyan
    ).astype(np.uint8)

    # ── Draw water boundary as bright cyan line ──
    from PIL import Image, ImageFilter
    mask_img  = Image.fromarray((mask_2d * 255).astype(np.uint8), mode='L')
    edges     = mask_img.filter(ImageFilter.FIND_EDGES)
    edge_arr  = np.array(edges) > 0
    overlay[edge_arr] = [0, 255, 255]   # bright cyan edge

    # ── Save as PNG ──
    img = Image.fromarray(overlay, mode='RGB')

    # Upscale for better visibility (128 → 512)
    img = img.resize((512, 512), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_metrics(pred, gt):
    pred = pred.flatten().astype(int)
    gt   = gt.flatten().astype(int)
    tp = ((pred==1)&(gt==1)).sum()
    fp = ((pred==1)&(gt==0)).sum()
    fn = ((pred==0)&(gt==1)).sum()
    precision = float(tp / (tp+fp+1e-8))
    recall    = float(tp / (tp+fn+1e-8))
    f1        = float(2*precision*recall / (precision+recall+1e-8))
    iou       = float(tp / (tp+fp+fn+1e-8))
    return {
        'IoU'      : round(iou,       4),
        'F1'       : round(f1,        4),
        'Precision': round(precision, 4),
        'Recall'   : round(recall,    4),
    }


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── 1. Read uploaded .tif file ──
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400

        file = request.files['image']
        ext  = file.filename.split('.')[-1].lower()

        if ext not in ['tif', 'tiff']:
            return jsonify({'error': 'Please upload a .tif or .tiff file'}), 400

        X_raw = read_tif(file.read())   # (H, W, C)

        # Validate shape
        if X_raw.shape != (128, 128, 12):
            return jsonify({
                'error': f'Expected shape (128, 128, 12), got {X_raw.shape}. '
                         f'Make sure you upload a single patch.'
            }), 400

        # Add batch dimension → (1, 128, 128, 12)
        X_raw = X_raw[np.newaxis, ...]
        payload = {'image': X_raw.tolist()}

        # ── 2. Call both model services in parallel ──
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fut_unet    = executor.submit(
                requests.post, UNET_URL,    json=payload, timeout=60)
            fut_deeplab = executor.submit(
                requests.post, DEEPLAB_URL, json=payload, timeout=60)
            unet_resp    = fut_unet.result()
            deeplab_resp = fut_deeplab.result()

        if unet_resp.status_code != 200:
            return jsonify({'error': 'U-Net service error: ' +
                           unet_resp.json().get('error', 'unknown')}), 500

        if deeplab_resp.status_code != 200:
            return jsonify({'error': 'DeepLab service error: ' +
                           deeplab_resp.json().get('error', 'unknown')}), 500

        unet_probs    = np.array(unet_resp.json()['probs'])     # (1,128,128,1)
        deeplab_probs = np.array(deeplab_resp.json()['probs'])  # (1,128,128,1)

        # ── 3. Ensemble ──
        ensemble_probs = (unet_probs + deeplab_probs) / 2

        # ── 4. Threshold → binary masks ──
        # unet_mask     = (unet_probs[0,:,:,0]     > BEST_THRESHOLD).astype(np.uint8)
        # deeplab_mask  = (deeplab_probs[0,:,:,0]  > BEST_THRESHOLD).astype(np.uint8)
        # ensemble_mask = (ensemble_probs[0,:,:,0] > BEST_THRESHOLD).astype(np.uint8)
        
        unet_mask     = (unet_probs[0,:,:,0]     > BEST_THRESHOLD).astype(np.uint8)
        deeplab_mask  = (deeplab_probs[0,:,:,0]  > BEST_THRESHOLD).astype(np.uint8)
        ensemble_mask = (ensemble_probs[0,:,:,0] > BEST_THRESHOLD).astype(np.uint8)

        # ── 5. Encode masks as base64 PNGs ──
        # results = {
        #     'unet_mask'    : mask_to_png_base64(unet_mask),
        #     'deeplab_mask' : mask_to_png_base64(deeplab_mask),
        #     'ensemble_mask': mask_to_png_base64(ensemble_mask),
        #     'water_pct': {
        #         'unet'    : round(float(unet_mask.mean())    * 100, 2),
        #         'deeplab' : round(float(deeplab_mask.mean()) * 100, 2),
        #         'ensemble': round(float(ensemble_mask.mean())* 100, 2),
        #     }
        # }

        # Raw single patch for RGB visualization
        X_single = X_raw[0]   # (128, 128, 12)

        results = {
            'unet_mask'    : mask_to_png_base64(X_single, unet_mask),
            'deeplab_mask' : mask_to_png_base64(X_single, deeplab_mask),
            'ensemble_mask': mask_to_png_base64(X_single, ensemble_mask),
            'water_pct': {
                'unet'    : round(float(unet_mask.mean())    * 100, 2),
                'deeplab' : round(float(deeplab_mask.mean()) * 100, 2),
                'ensemble': round(float(ensemble_mask.mean())* 100, 2),
            }
        }

        # ── 6. Ground truth mask (optional) ──
        if 'mask' in request.files:
            mask_file = request.files['mask']
            mask_ext  = mask_file.filename.split('.')[-1].lower()

            if mask_ext in ['tif', 'tiff']:
                gt = read_tif(mask_file.read(), is_mask=True)    # (H, W)

            elif mask_ext == 'png':
                from PIL import Image
                img = Image.open(io.BytesIO(mask_file.read()))
                gt  = np.array(img)

                # Handle grayscale, RGB, RGBA
                if gt.ndim == 3:
                    gt = gt[:, :, 0]     # take first channel

                # Binarize — 0/255 PNG → 0/1
                gt = (gt > 0).astype(np.uint8)

            elif mask_ext == 'npy':
                gt = np.load(io.BytesIO(mask_file.read()))
                gt = gt.squeeze()
                gt = (gt > 0).astype(np.uint8)

            else:
                return jsonify({'error': f'Unsupported mask format: .{mask_ext}'}), 400

            # Verify shape
            if gt.shape != (128, 128):
                return jsonify({
                    'error': f'Mask shape mismatch. Expected (128, 128), got {gt.shape}'
                }), 400

            results['metrics'] = {
                'U-Net'     : compute_metrics(unet_mask,     gt),
                'DeepLabV3+': compute_metrics(deeplab_mask,  gt),
                'Ensemble'  : compute_metrics(ensemble_mask, gt),
            }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    try:
        unet_ok    = requests.get(UNET_HEALTH,    timeout=3).ok
        deeplab_ok = requests.get(DEEPLAB_HEALTH, timeout=3).ok
    except:
        unet_ok = deeplab_ok = False
    return jsonify({
        'flask_app': 'ok',
        'unet'     : 'ok' if unet_ok    else 'down',
        'deeplab'  : 'ok' if deeplab_ok else 'down',
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)