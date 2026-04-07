import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.ml.lung_model import predict, get_gradcam
from backend.ml.clip_filter import is_valid
from backend.utils.image_utils import b64_to_pil, ndarray_to_b64
from backend.utils.plot_utils import prob_chart_to_b64


def process_lung(image_b64: str) -> dict:
    image = b64_to_pil(image_b64)

    valid, score = is_valid(image, modality="lung", threshold=0.40)
    if not valid:
        raise ValueError(
            f"Yüklənən şəkil ağciyər rentgeni deyil (uyğunluq: {score*100:.0f}%). "
            f"Zəhmət olmasa düzgün ağciyər rentgeni şəkli yükləyin."
        )

    result  = predict(image)
    gradcam = get_gradcam(image)

    return {
        "label":           result["label"],
        "confidence":      result["confidence"],
        "probabilities":   result["probabilities"],
        "heatmap_b64":     ndarray_to_b64(gradcam),
        "prob_chart_b64":  prob_chart_to_b64(result["probabilities"], "Chest X-Ray Result"),
    }