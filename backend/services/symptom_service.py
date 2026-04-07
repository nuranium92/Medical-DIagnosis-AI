import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.ml.symptom_checker import predict
from backend.utils.plot_utils import shap_chart_to_b64


def process_symptom(text: str) -> dict:
    result = predict(text)

    if "error" in result:
        return {
            "matched_symptoms": [],
            "predictions":      [],
            "explanation":      [],
            "shap_chart_b64":   "",
            "llm_summary":      "",
            "low_confidence":   False,
            "error":            result["error"],
        }

    shap_b64 = shap_chart_to_b64(result["explanation"], "Symptom Impact (SHAP)")

    return {
        "matched_symptoms": result["matched_symptoms"],
        "predictions":      result["predictions"],
        "explanation":      result["explanation"],
        "shap_chart_b64":   shap_b64,
        "llm_summary":      result.get("llm_summary", ""),
        "low_confidence":   result.get("low_confidence", False),
        "error":            None,
    }