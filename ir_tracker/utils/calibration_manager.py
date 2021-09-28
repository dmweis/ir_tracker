import cv2
import numpy as np
import yaml

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


class ImageCalibration:
    def __init__(self, mtx, camera_matrix, dist, rvecs, tvecs, roi):
        self.mtx = mtx
        self.camera_matrix = camera_matrix
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.roi = roi

    def undistort_image(self, image, roi_only=False):
        dst = cv2.undistort(image, self.mtx, self.dist, None,
                            self.camera_matrix)
        if roi_only:
            x, y, w, h = self.roi
            dst = dst[y:y + h, x:x + w]
        return dst

    def save_yaml(self, path):
        calibration_data = {}
        calibration_data["mtx"] = self.mtx.tolist()
        calibration_data["camera_matrix"] = self.camera_matrix.tolist()
        calibration_data["dist"] = self.dist.tolist()
        calibration_data["roi"] = list(self.roi)
        calibration_data["rvecs"] = [item.tolist() for item in self.rvecs]
        calibration_data["tvecs"] = [item.tolist() for item in self.tvecs]

        with open(path, 'w') as yaml_file:
            yaml.dump(calibration_data, yaml_file)

    @staticmethod
    def load_yaml(path):
        with open(path, 'r') as yaml_file:
            calibration = yaml.safe_load(yaml_file)
            mtx = np.asarray(calibration["mtx"], dtype=np.float32)
            camera_matrix = np.asarray(calibration["camera_matrix"],
                                       dtype=np.float32)
            dist = np.asarray(calibration["dist"], dtype=np.float32)
            roi = calibration["roi"]

            rvecs = [
                np.asarray(rvec, dtype=np.float32)
                for rvec in calibration["rvecs"]
            ]
            tvecs = [
                np.asarray(tvec, dtype=np.float32)
                for tvec in calibration["tvecs"]
            ]
            return ImageCalibration(mtx, camera_matrix, dist, rvecs, tvecs,
                                    roi)


def calibarate_from_images(images,
                           chessboard_height=7,
                           chessboard_width=6,
                           square_size=1.0,
                           display_time=-1):
    chessboard_coordinates = np.zeros(
        (chessboard_width * chessboard_height, 3), np.float32)
    chessboard_coordinates[:, :2] = np.mgrid[0:chessboard_height,
                                             0:chessboard_width].T.reshape(
                                                 -1, 2)
    chessboard_coordinates *= square_size
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray, (chessboard_height, chessboard_width), None)
        if ret == True:
            objpoints.append(chessboard_coordinates)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        CRITERIA)
            imgpoints.append(corners2)

            if display_time != -1:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(
                    image, (chessboard_height, chessboard_width), corners2,
                    ret)
                cv2.imshow('img', img)
                cv2.waitKey(display_time)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray.shape[::-1], None,
                                                       None)
    h, w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1,
                                                      (w, h))
    return ImageCalibration(mtx, newcameramtx, dist, rvecs, tvecs, roi)


def draw_pose(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def display_chessboard_pose(image,
                            calibration,
                            chessboard_height=7,
                            chessboard_width=6,
                            display_time=-1):
    mtx = calibration.mtx
    dist = calibration.dist
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboard_width * chessboard_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_height,
                           0:chessboard_width].T.reshape(-1, 2)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, (chessboard_height, chessboard_width), None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria)
        # ret, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        image = draw_pose(image, corners2, imgpts)
    if display_time > 0:
        cv2.imshow("Image", image)
        cv2.waitKey(display_time)
