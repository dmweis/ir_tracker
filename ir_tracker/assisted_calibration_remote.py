import glob
import time
from os import read
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt

from ir_tracker.utils import calibration_manager, utility


def draw_info(image, text):
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2, cv2.LINE_AA)


CHESSBOARD_HEIGHT = 10
CHESSBOARD_WIDTH = 7
PICTURE_TIME = 3
NUMBER_OF_IMAGES = 10
# record images
calibration_images = []

while len(calibration_images) < 10:
    start_time = time.time()
    while True:
        current_time = time.time()
        time_delta = current_time - start_time
        if time_delta > PICTURE_TIME:
            image = utility.request_image()
            clean_image = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found_chessboard, corners = cv2.findChessboardCorners(
                gray, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), None)
            cv2.drawChessboardCorners(image,
                                      (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH),
                                      corners, found_chessboard)
            if found_chessboard:
                calibration_images.append(clean_image)
                draw_info(image, "Image saved")
                cv2.imshow('img', image)
                cv2.waitKey(1000)
            else:
                draw_info(image, "Chessboard not found")
                cv2.imshow('img', image)
                cv2.waitKey(1000)
            break
        image = read_image()
        draw_info(
            image,
            f"{PICTURE_TIME - time_delta:.1f}s left, {len(calibration_images)}/{NUMBER_OF_IMAGES}"
        )
        # detect chessboard
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found_chessboard, corners = cv2.findChessboardCorners(
            gray, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), None)
        cv2.drawChessboardCorners(image, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH),
                                  corners, found_chessboard)
        cv2.imshow('img', image)
        cv2.waitKey(1)

# image_paths = glob.glob('sample_calibration_images/*.jpg')
# calibration_images = [cv2.imread(image_path) for image_path in image_paths]

image_directory = "calibration_images"
Path(image_directory).mkdir(parents=True, exist_ok=True)
print(f"Saving images to {image_directory}")
for i, image in enumerate(calibration_images):
    cv2.imwrite(f"{image_directory}/image_{i}.png", image)
print("images saved")

print("Calibrating")
calibartion = calibration_manager.calibarate_from_images(
    calibration_images, CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH, 500)

Path("calibration").mkdir(parents=True, exist_ok=True)
calibration_path = "calibration/picamera_calibration.yml"
print(f"Saving calibration to {calibration_path}")
calibartion.save_yaml(calibration_path)
calibartion_read = calibration_manager.ImageCalibration.load_yaml(
    calibration_path)

for image in calibration_images:
    undisorted = calibartion_read.undistort_image(image, False)
    combined = cv2.vconcat((image, undisorted))
    cv2.imshow('Image', combined)
    cv2.waitKey(500)

cv2.destroyAllWindows()
