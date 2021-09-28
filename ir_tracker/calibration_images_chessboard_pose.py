import cv2
import numpy as np
from ir_tracker.utils import calibration_manager
import glob

calibartion_read = calibration_manager.ImageCalibration.load_yaml(
    "calibration/picamera_calibration.yml")
mtx = calibartion_read.mtx
dist = calibartion_read.dist


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


CHESSBOARD_HEIGHT = 10
CHESSBOARD_WIDTH = 7

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARD_WIDTH * CHESSBOARD_HEIGHT, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_HEIGHT,
                       0:CHESSBOARD_WIDTH].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

image_paths = glob.glob('calibration_images/*.png')
for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria)
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
    cv2.imshow("img", img)
    cv2.waitKey(800)

cv2.destroyAllWindows()
