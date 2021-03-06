from urllib.request import urlopen

import cv2
import numpy as np

from ir_tracker.utils import calibration_manager, utility

calibartion_read = calibration_manager.ImageCalibration.load_yaml(
    "calibration/picamera_calibration.yml")
mtx = calibartion_read.mtx
dist = calibartion_read.dist

# object_points = np.float32([[0, 0, 0], [0.12, 0, 0], [0.06, -0.12, 0],
#                             [0.06 + 0.056, -0.12 - 0.056, 0]])
object_points = np.zeros((3 * 4, 3), np.float32)
object_points[:, :2] = np.mgrid[0:4, 0:3].T.reshape(-1, 2)
# object_points = np.delete(object_points, 4, 0)
# print("object points\n", object_points)
# exit()
axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
use_ransac = False


def draw_pose(img, corners, imgpts):
    try:
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img
    except:
        pass


if __name__ == "__main__":
    # for image_id in range(10):
    use_otsu_thresholding = False
    binarization_threshold = 180
    while True:
        # image = cv2.imread(f"calibration_images/image000{image_id}.jpg")
        image = utility.request_image()
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if use_otsu_thresholding:
            threshold_value, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print("Thresholding with", threshold_value)
        else:
            threshold_value, thresh = cv2.threshold(gray,
                                                    binarization_threshold,
                                                    255, cv2.THRESH_BINARY)

        cnts_individual = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        # this is some weirdness where the return can be a tuple of 3 or 4 elements
        cnts_individual = cnts_individual[0] if len(
            cnts_individual) == 2 else cnts_individual[1]
        image_points = []
        for c in cnts_individual:
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(c)

            # Find centroid
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                image_points.append([[cX], [cY]])

                # Draw the contour and center of the shape on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(image, (cX, cY), 1, (320, 159, 22), 8)
                cv2.putText(image, '({}, {})'.format(cX, cY), (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            except ZeroDivisionError:
                pass
        image_points = np.float32(image_points)
        if len(image_points) == 12:
            if use_ransac:
                ret, rvecs, tvecs, _ = cv2.solvePnPRansac(
                    object_points, image_points, mtx, dist)
            else:
                ret, rvecs, tvecs = cv2.solvePnP(object_points, image_points,
                                                 mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            original = draw_pose(original, image_points, imgpts)

        colored_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        try:
            combined = cv2.vconcat((original, image))
            cv2.imshow("thresh", combined)
            cv2.waitKey(50)
        except:
            pass
        # while 113 != cv2.waitKey(5):
        #     pass
