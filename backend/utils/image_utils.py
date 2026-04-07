import base64
import io
import numpy as np
from PIL import Image


def pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ndarray_to_b64(arr: np.ndarray, fmt: str = "PNG") -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    return pil_to_b64(img, fmt)


def b64_to_pil(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))