import numpy as np
import base64
import cv2


def image_from_bytes(img_byte):
    jpg_original = base64.b64decode(img_byte)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=1)
    return frame


def convert_images_to_base64_str(image, quality=90) -> str:
    assert (
        isinstance(quality, int) and 0 <= quality <= 100
    ), "Quality value must be an integer between 0 and 100."

    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    base64_image_str = base64.b64encode(buffer).decode("utf-8")
    return base64_image_str
