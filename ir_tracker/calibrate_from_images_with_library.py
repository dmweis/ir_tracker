import cv2
import glob
import numpy as np
import yaml
from ir_tracker.utils import calibration_manager

CHESSBOARD_HEIGHT = 10
CHESSBOARD_WIDTH = 7

image_paths = glob.glob('calibration_images/*.png')
images = [cv2.imread(image_path) for image_path in image_paths]

calibartion = calibration_manager.calibarate_from_images(
    images, CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH, 500)

calibartion.save_yaml("calibration/calibration.yml")
calibartion_read = calibration_manager.ImageCalibration.load_yaml(
    "calibration/calibration.yml")

for image_path in image_paths:
    img = cv2.imread(image_path)
    undisorted = calibartion_read.undistort_image(img, False)
    combined = cv2.vconcat((img, undisorted))
    cv2.imshow('img', combined)
    cv2.waitKey(500)

cv2.destroyAllWindows()