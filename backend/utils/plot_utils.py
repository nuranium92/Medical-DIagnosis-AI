import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backend.utils.image_utils import ndarray_to_b64
from PIL import Image


def prob_chart_to_b64(probabilities: dict, title: str = "Probabilities") -> str:
    labels = list(probabilities.keys())
    values = [probabilities[k] * 100 for k in labels]

    color_map = {
        "NORMAL":          "#2ecc71",
        "PNEUMONIA":       "#e74c3c",
        "No Tumor":        "#2ecc71",
        "Glioma":          "#e74c3c",
        "Meningioma":      "#f39c12",
        "Pituitary Tumor": "#9b59b6",
    }
    colors = [color_map.get(l, "#3498db") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Confidence (%)")
    ax.set_title(title, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=110, bbox_inches="tight")
    buf.seek(0)
    plt.close()

    arr = np.array(Image.open(buf))
    return ndarray_to_b64(arr)


def shap_chart_to_b64(explanation: list, title: str = "Symptom Impact (SHAP)") -> str:
    if not explanation:
        return ""

    symptoms = [e["symptom"] for e in explanation]
    impacts  = [e["impact"]  for e in explanation]
    colors   = ["#e74c3c" if v > 0 else "#3498db" for v in impacts]

    fig, ax = plt.subplots(figsize=(9, max(4, len(symptoms) * 0.5 + 2)))
    bars = ax.barh(symptoms, impacts, color=colors, edgecolor="white", height=0.6)

    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Impact")
    ax.set_title(title, fontsize=13)

    for bar, val in zip(bars, impacts):
        ax.text(
            val + (0.05 if val >= 0 else -0.05),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=110, bbox_inches="tight")
    buf.seek(0)
    plt.close()

    arr = np.array(Image.open(buf))
    return ndarray_to_b64(arr)