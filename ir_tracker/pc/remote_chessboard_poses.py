from urllib.request import urlopen

import cv2
import numpy as np

from ir_tracker.utils import calibration_manager, utility

calibartion_read = calibration_manager.ImageCalibration.load_yaml(
    "calibration/picamera_calibration.yml")

CHESSBOARD_HEIGHT = 10
CHESSBOARD_WIDTH = 7

while True:
    image = utility.request_image()
    calibration_manager.display_chessboard_pose(image, calibartion_read,
                                                CHESSBOARD_HEIGHT,
                                                CHESSBOARD_WIDTH, 20)
