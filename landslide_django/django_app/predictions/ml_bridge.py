"""
django_app/predictions/ml_bridge.py

Bridges Django to the mhscience/landslides_detection ML pipeline.
Runs: image preprocessing → feature extraction → Random Forest prediction
→ generates visualizations (heatmap, feature chart, segmentation overlay).

The ML pipeline from the original repo uses these six features:
  ndvi, b3, slope_mean, brightness, ndvi_change, ratio_rg_change

If a trained model (model/rf_model.pkl) exists in the cloned repo,
it will be loaded. Otherwise a demo model is trained on synthetic data
so the Django app still runs end-to-end for testing.
"""

import os
import sys
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from django.conf import settings


# ─── Feature column names (match the original repo) ─────────────────────────
FEATURE_COLS = ['ndvi', 'b3', 'slope_mean', 'brightness', 'ndvi_change', 'ratio_rg_change']


# ─── Locate / load the trained Random Forest model ───────────────────────────
def _load_model():
    """Load the RF model from the cloned repo, or create a demo model."""
    ml_path = Path(getattr(settings, 'ML_REPO_PATH', ''))
    model_candidates = [
        ml_path / 'model' / 'rf_model.pkl',
        ml_path / 'model' / 'model.pkl',
        ml_path / 'trained_model.pkl',
    ]
    for candidate in model_candidates:
        if candidate.exists():
            print(f"[ml_bridge] Loading model from: {candidate}")
            return joblib.load(candidate)

    # ── Demo model: train on small synthetic dataset ─────────────────────────
    print("[ml_bridge] No trained model found — using demo Random Forest.")
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(42)

    # Class 0: non-landslide — high NDVI, low slope change
    X0 = rng.normal(loc=[0.6, 0.4, 8,  0.45, -0.02, 1.05], scale=[0.1, 0.05, 3,  0.05, 0.02, 0.05], size=(200, 6))
    # Class 1: landslide   — low NDVI,  high slope change
    X1 = rng.normal(loc=[0.1, 0.6, 25, 0.60,  0.25, 1.40], scale=[0.1, 0.05, 5,  0.05, 0.05, 0.1],  size=(200, 6))
    X  = np.vstack([X0, X1])
    y  = np.array([0]*200 + [1]*200)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf


_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


# ─── Feature extraction from image ───────────────────────────────────────────
def extract_features_from_image(image_path: str) -> dict:
    """
    Extract spectral features from a satellite/regular image.
    Returns estimated values for the six features the RF expects.
    In production, replace this with actual band extraction from
    a multi-band GeoTIFF (rasterio) or GEE-exported file.
    """
    from PIL import Image as PILImage

    img = PILImage.open(image_path).convert('RGB')
    img_arr = np.array(img, dtype=np.float32) / 255.0   # normalise to 0-1

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    # Approximate NDVI from R and G (true NDVI needs NIR band)
    ndvi = float(np.mean((g - r) / (g + r + 1e-8)))

    # Band 3 proxy (Sentinel-2 B3 = Green)
    b3 = float(np.mean(g))

    # Brightness = mean of all channels
    brightness = float(np.mean(img_arr))

    # Slope proxy: local variance in brightness as texture proxy for terrain roughness
    from scipy.ndimage import uniform_filter
    local_std = float(np.std(uniform_filter(brightness * np.ones_like(r), size=5)))
    slope_mean = max(2.0, local_std * 80)  # scaled to plausible degree range

    # Synthetic change values (set to 0 if no pre-image is provided)
    ndvi_change    = 0.0
    ratio_rg_change = float(np.mean(r / (g + 1e-8)))

    return {
        'ndvi':          round(ndvi,          4),
        'b3':            round(b3,            4),
        'slope_mean':    round(slope_mean,    4),
        'brightness':    round(brightness,    4),
        'ndvi_change':   round(ndvi_change,   4),
        'ratio_rg_change': round(ratio_rg_change, 4),
    }


# ─── Run prediction ───────────────────────────────────────────────────────────
def run_prediction(features: dict) -> dict:
    """
    Given a dict of the six feature values, run the Random Forest
    and return probability, class, and feature importances.
    """
    model = get_model()
    X = np.array([[features.get(f, 0.0) for f in FEATURE_COLS]])
    prob_arr   = model.predict_proba(X)[0]
    pred_class = int(model.predict(X)[0])
    probability = float(prob_arr[1]) if len(prob_arr) > 1 else float(prob_arr[0])

    # Feature importances
    importances = {}
    if hasattr(model, 'feature_importances_'):
        for name, imp in zip(FEATURE_COLS, model.feature_importances_):
            importances[name] = round(float(imp), 4)

    return {
        'predicted_class':      pred_class,
        'landslide_probability': round(probability, 4),
        'confidence_score':     round(max(prob_arr), 4),
        'feature_importance':   importances,
    }


# ─── Determine risk level ─────────────────────────────────────────────────────
def probability_to_risk(probability: float) -> str:
    if probability < 0.35:  return 'low'
    if probability < 0.55:  return 'moderate'
    if probability < 0.75:  return 'high'
    return 'critical'


# ─── Generate visualizations ─────────────────────────────────────────────────
def generate_heatmap(image_path: str, probability: float, out_path: str):
    """Overlay a colour-coded risk heatmap on the original image."""
    from PIL import Image as PILImage

    img = PILImage.open(image_path).convert('RGB')
    img_arr = np.array(img, dtype=np.float32) / 255.0

    # Create a heat overlay based on local brightness as risk proxy
    gray = np.mean(img_arr, axis=2)
    heat = (1 - gray) * probability        # darker regions → higher risk proxy

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')

    axes[0].imshow(img_arr)
    axes[0].set_title('Original Image', color='white', fontsize=13, pad=10)
    axes[0].axis('off')

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'risk', ['#22c55e', '#f59e0b', '#ef4444', '#7c3aed']
    )
    im = axes[1].imshow(heat, cmap=cmap, vmin=0, vmax=1, alpha=0.85)
    axes[1].imshow(img_arr, alpha=0.45)
    axes[1].set_title('Risk Heatmap', color='white', fontsize=13, pad=10)
    axes[1].axis('off')

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Risk Level', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)


def generate_feature_chart(features: dict, importances: dict, out_path: str):
    """Bar chart showing feature values and their importance."""
    labels   = FEATURE_COLS
    values   = [features.get(f, 0.0)    for f in labels]
    imp_vals = [importances.get(f, 0.0) for f in labels]

    nice_labels = ['NDVI', 'Band 3\n(Green)', 'Slope\nMean', 'Brightness', 'NDVI\nChange', 'RG Ratio\nChange']
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')

    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        ax.spines[:].set_color('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=9)

    # Feature values
    bars1 = ax1.bar(nice_labels, values, color=colors, edgecolor='#21262d', linewidth=0.8)
    ax1.set_title('Extracted Feature Values', color='white', fontsize=12, pad=12)
    ax1.set_ylabel('Value', color='#8b949e', fontsize=10)
    for bar, val in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=8)

    # Feature importance
    bars2 = ax2.barh(nice_labels[::-1], imp_vals[::-1], color=colors[::-1],
                     edgecolor='#21262d', linewidth=0.8)
    ax2.set_title('Feature Importance (Random Forest)', color='white', fontsize=12, pad=12)
    ax2.set_xlabel('Importance', color='#8b949e', fontsize=10)
    for bar, val in zip(bars2, imp_vals[::-1]):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', color='white', fontsize=8)

    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)


def generate_segmentation_overlay(image_path: str, probability: float, out_path: str):
    """Simple K-means segmentation overlay (3 clusters) with risk colour coding."""
    from PIL import Image as PILImage
    from sklearn.cluster import MiniBatchKMeans

    img = PILImage.open(image_path).convert('RGB').resize((256, 256))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    h, w, _ = img_arr.shape
    pixels  = img_arr.reshape(-1, 3)

    kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=3)
    labels = kmeans.fit_predict(pixels).reshape(h, w)

    # Map cluster with darkest centroid to "landslide" if probability > 0.5
    cluster_brightness = kmeans.cluster_centers_.mean(axis=1)
    darkest_cluster    = int(np.argmin(cluster_brightness))
    risk_mask          = (labels == darkest_cluster).astype(float) * probability

    cmap_seg  = mcolors.LinearSegmentedColormap.from_list('seg', ['#1e3a5f', '#3b82f6', '#93c5fd'])
    cmap_risk = mcolors.LinearSegmentedColormap.from_list('risk', ['#00000000', '#ef4444cc'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')
    titles = ['Original', 'Segmentation', 'Risk Overlay']

    axes[0].imshow(img_arr)
    axes[1].imshow(labels, cmap=cmap_seg)
    axes[2].imshow(img_arr)
    axes[2].imshow(risk_mask, cmap=cmap_risk, alpha=0.7, vmin=0, vmax=1)

    for ax, t in zip(axes, titles):
        ax.set_title(t, color='white', fontsize=12, pad=10)
        ax.axis('off')

    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)


# ─── Full pipeline entry point ────────────────────────────────────────────────
def run_full_pipeline(analysis_obj):
    """
    Main entry point called by Django view.
    Accepts a LandslideAnalysis model instance, runs the full ML pipeline,
    saves visualisations, and updates the model instance in-place.
    Returns the updated instance.
    """
    from django.core.files.base import ContentFile
    import traceback

    try:
        image_path = analysis_obj.image.path

        # 1. Extract features from image (supplement with user-provided values)
        auto_features = extract_features_from_image(image_path)
        features = {
            'ndvi':           analysis_obj.ndvi          if analysis_obj.ndvi          is not None else auto_features['ndvi'],
            'b3':             analysis_obj.b3             if analysis_obj.b3             is not None else auto_features['b3'],
            'slope_mean':     analysis_obj.slope_mean     if analysis_obj.slope_mean     is not None else auto_features['slope_mean'],
            'brightness':     analysis_obj.brightness     if analysis_obj.brightness     is not None else auto_features['brightness'],
            'ndvi_change':    analysis_obj.ndvi_change    if analysis_obj.ndvi_change    is not None else auto_features['ndvi_change'],
            'ratio_rg_change':analysis_obj.ratio_rg_change if analysis_obj.ratio_rg_change is not None else auto_features['ratio_rg_change'],
        }

        # Store extracted features back
        for k, v in features.items():
            setattr(analysis_obj, k, v)

        # 2. Run Random Forest prediction
        result = run_prediction(features)
        analysis_obj.predicted_class        = result['predicted_class']
        analysis_obj.landslide_probability  = result['landslide_probability']
        analysis_obj.confidence_score       = result['confidence_score']
        analysis_obj.risk_level             = probability_to_risk(result['landslide_probability'])
        analysis_obj.feature_importance     = result['feature_importance']

        # 3. Generate visualisations
        media_root = Path(settings.MEDIA_ROOT)
        vis_dir    = media_root / 'results' / str(analysis_obj.pk)
        vis_dir.mkdir(parents=True, exist_ok=True)

        heatmap_path    = vis_dir / 'heatmap.png'
        chart_path      = vis_dir / 'feature_chart.png'
        seg_path        = vis_dir / 'segmentation.png'

        generate_heatmap(image_path, result['landslide_probability'], str(heatmap_path))
        generate_feature_chart(features, result['feature_importance'], str(chart_path))
        generate_segmentation_overlay(image_path, result['landslide_probability'], str(seg_path))

        # Save relative paths to model
        rel = lambda p: str(p.relative_to(media_root))
        analysis_obj.heatmap_image      = rel(heatmap_path)
        analysis_obj.feature_chart_image= rel(chart_path)
        analysis_obj.segmentation_image = rel(seg_path)

        analysis_obj.status = 'completed'

    except Exception as exc:
        analysis_obj.status        = 'failed'
        analysis_obj.error_message = traceback.format_exc()
        print(f"[ml_bridge] Pipeline failed: {exc}")

    analysis_obj.save()
    return analysis_obj