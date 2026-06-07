"""
src/ml/image_model.py — EfficientNet-B0 mammogram image classifier + Grad-CAM.

Model weights (efficientnet_cbis.pth) are trained on CBIS-DDSM via Kaggle GPU
and placed in models/ folder. Inference runs on CPU locally.

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
    """True if model weights file exists and all dependencies are installed."""
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


def predict_image(pil_image) -> dict:
    """
    Run EfficientNet-B0 on a PIL image and return risk score + Grad-CAM overlay.

    Parameters
    ----------
    pil_image : PIL.Image.Image

    Returns
    -------
    dict:
        probability       float — P(malignant), 0–1
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

    # Grad-CAM — use conv_head (last conv layer before pooling in EfficientNet-B0)
    cam_b64 = ""
    if _CAM_OK:
        try:
            target_layer = model.conv_head
            cam = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=tensor, targets=None)[0]

            rgb_arr = np.array(img_rgb.resize((224, 224))).astype(np.float32) / 255.0
            overlay = show_cam_on_image(rgb_arr, grayscale_cam, use_rgb=True)

            buf = io.BytesIO()
            PILImage.fromarray(overlay).save(buf, format="PNG")
            cam_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception as exc:
            logger.warning("Grad-CAM failed: %s", exc)

    pct = round(prob * 100, 1)
    if prob >= 0.65:
        risk = "Elevated"
    elif prob >= 0.35:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        "probability":     prob,
        "risk_category":   risk,
        "confidence_band": f"{pct}% ± 7%",
        "gradcam_png_b64": cam_b64,
        "model_name":      "EfficientNet-B0 / CBIS-DDSM",
        "gradcam_available": bool(cam_b64),
    }
