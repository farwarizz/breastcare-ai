"""
src/ml/image_model.py — EfficientNet-B0 mammogram image classifier + Grad-CAM.

Model weights (efficientnet_cbis.pth) were trained on CBIS-DDSM on a Kaggle GPU
and placed in models/. Inference runs on CPU.

Experimental: no evaluation results for this model are stored in this repo,
so its accuracy is unverified. There is deliberately no fallback — if the
weights or torch/timm are missing, the API returns HTTP 503 instead of a score.

Usage:
    from src.ml.image_model import predict_image, is_available
    if is_available():
        result = predict_image(pil_image)
"""
from __future__ import annotations
import io, base64, logging
from functools import lru_cache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Deferred imports — torch/timm not required if model weights aren't present
_TORCH_OK = False
try:
    import torch
    import numpy as np
    from PIL import Image as PILImage
    _TORCH_OK = True
except ImportError:
    pass

_TIMM_OK = False
try:
    import timm
    _TIMM_OK = True
except ImportError:
    pass

_CAM_OK = False
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _CAM_OK = True
except ImportError:
    pass

try:
    from torchvision import transforms as T
    _TV_OK = True
except ImportError:
    _TV_OK = False

from src.config import MODELS_DIR

IMG_MODEL_PATH = MODELS_DIR / "efficientnet_cbis.pth"

_TRANSFORM = None
if _TORCH_OK and _TV_OK:
    _TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def is_available() -> bool:
    """True only if the real weights file exists and torch/torchvision/timm import."""
    return (
        _TORCH_OK and _TIMM_OK and _TV_OK
        and IMG_MODEL_PATH.exists()
    )


@lru_cache(maxsize=1)
def _load_model():
    if not _TORCH_OK or not _TIMM_OK:
        raise RuntimeError("torch and timm required — install with: pip install torch torchvision timm")
    if not IMG_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Image model weights not found at {IMG_MODEL_PATH}. "
            "Train on Kaggle and place efficientnet_cbis.pth in models/"
        )
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    state = torch.load(str(IMG_MODEL_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("EfficientNet-B0 image model loaded from %s", IMG_MODEL_PATH)
    return model


def ensure_loaded() -> None:
    """Load (and cache) the weights; raises if they cannot be loaded."""
    _load_model()


def predict_image(pil_image) -> dict:
    """
    Run EfficientNet-B0 on a PIL image and return risk score + Grad-CAM overlay.

    Parameters
    ----------
    pil_image : PIL.Image.Image

    Returns
    -------
    dict:
        probability       float — model score for P(malignant), 0–1; uncalibrated,
                                  no uncertainty interval
        calibrated        bool  — always False
        risk_category     str   — Low / Moderate / Elevated
        gradcam_png_b64   str   — base64-encoded PNG of Grad-CAM overlay
        model_name        str   — "EfficientNet-B0 / CBIS-DDSM"
    """
    if not is_available():
        raise RuntimeError("Image model not available — weights missing or torch not installed")

    model = _load_model()
    img_rgb = pil_image.convert("RGB")
    tensor = _TRANSFORM(img_rgb).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)
        prob = float(torch.softmax(logits, dim=1)[0, 1])

    # Grad-CAM — manual implementation using PyTorch hooks
    cam_b64 = ""
    if _TORCH_OK:
        try:
            # Hook into conv_head to capture activations and gradients
            activations = {}
            gradients = {}

            def fwd_hook(module, inp, out):
                activations['value'] = out.detach()

            def bwd_hook(module, grad_in, grad_out):
                gradients['value'] = grad_out[0].detach()

            target_layer = model.conv_head
            fh = target_layer.register_forward_hook(fwd_hook)
            bh = target_layer.register_full_backward_hook(bwd_hook)

            # Forward + backward for malignant class
            tensor_cam = tensor.clone().requires_grad_(True)
            output = model(tensor_cam)
            model.zero_grad()
            output[0, 1].backward()  # class 1 = malignant

            fh.remove()
            bh.remove()

            # Compute Grad-CAM
            grads = gradients['value']        # (1, C, H, W)
            acts = activations['value']       # (1, C, H, W)
            weights = grads.mean(dim=[2, 3], keepdim=True)  # GAP over spatial dims
            cam_map = torch.relu((weights * acts).sum(dim=1, keepdim=True))
            cam_map = cam_map.squeeze().numpy()
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

            # Resize to 224×224 and overlay on original image
            from PIL import Image as PILImg
            cam_resized = np.array(PILImg.fromarray((cam_map * 255).astype(np.uint8)).resize((224, 224))) / 255.0
            rgb_arr = np.array(img_rgb.resize((224, 224))).astype(np.float32) / 255.0

            # Create heatmap overlay
            import colorsys
            heatmap = np.zeros((*cam_resized.shape, 3), dtype=np.float32)
            for i in range(cam_resized.shape[0]):
                for j in range(cam_resized.shape[1]):
                    v = cam_resized[i, j]
                    r, g, b = colorsys.hsv_to_rgb(max(0, (1 - v) * 0.6), 1.0, 1.0)
                    heatmap[i, j] = [r, g, b]

            overlay = np.clip(rgb_arr * 0.5 + heatmap * 0.5, 0, 1)
            overlay_img = PILImg.fromarray((overlay * 255).astype(np.uint8))

            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            cam_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception as exc:
            logger.warning("Grad-CAM failed: %s", exc)

    if prob >= 0.65:
        risk = "Elevated"
    elif prob >= 0.35:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        "probability":     prob,
        "calibrated":      False,
        "risk_category":   risk,
        "gradcam_png_b64": cam_b64,
        "model_name":      "EfficientNet-B0 / CBIS-DDSM",
        "gradcam_available": bool(cam_b64),
    }
