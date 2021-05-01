import cv2
import numpy as np
from ir_tracker import calibration_manager
from urllib.request import urlopen


def read_image():
    resp = urlopen('http://camerapi.local:8080/?action=snapshot')
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


calibartion_read = calibration_manager.ImageCalibration.load_yaml(
    "calibration/picamera_calibration.yml")

while True:
    image = read_image()
    calibration_manager.display_chessboard_pose(image, calibartion_read, 9, 6,
                                                20)
