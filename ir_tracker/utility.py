from urllib.request import urlopen

import cv2
import numpy as np


def request_image(url: str = "http://camerapi2.local:8080/?action=snapshot"):
    resp = urlopen('http://camerapi2.local:8080/?action=snapshot')
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # width, height = image.shape[:2]
    return image


def concat_images(image_grid):
    rows = []
    for row_images in image_grid:
        if isinstance(row_images, list):
            combined_row = cv2.hconcat(row_images)
            rows.append(combined_row)
        else:
            rows.append(row_images)
    combined = cv2.vconcat(rows)
    return combined


def threshold_image(image, binarization_threshold=180, use_otsu: bool = False):
    """
    Returns:
        (image, selected_threshold)
    """
    if use_otsu:
        threshold_value, thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (thresh, threshold_value)
    threshold_value, thresh = cv2.threshold(image, binarization_threshold, 255,
                                            cv2.THRESH_BINARY)
    return (thresh, threshold_value)