import cv2
import numpy as np
from ir_tracker.utils import utility

if __name__ == "__main__":
    counter = utility.FramerateCounter()
    use_otsu_thresholding = False
    binarization_threshold = 180
    while True:
        # image = cv2.imread(f"calibration_images/image000{image_id}.jpg")
        image = utility.request_image(
            url="http://camerapi2.local:8000/image_frame/last_image")

        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh, threshold_value = utility.threshold_image(
            gray, binarization_threshold, use_otsu_thresholding)

        cnts_individual = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        # this is some weirdness where the return can be a tuple of 3 or 4 elements
        cnts_individual = cnts_individual[0] if len(
            cnts_individual) == 2 else cnts_individual[1]
        print("Found", len(cnts_individual), "points")
        for c in cnts_individual:
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(c)

            # Find centroid
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the contour and center of the shape on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(image, (cX, cY), 1, (320, 159, 22), 8)
                cv2.putText(image, '({}, {})'.format(cX, cY), (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            except ZeroDivisionError:
                pass

        colored_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # combined = cv2.vconcat((original, image))
        combined = utility.concat_images([original, colored_thresh, image])
        print("framerate", counter.measure_fps())
        cv2.imshow("thresh", combined)
        cv2.waitKey(50)
        # while 113 != cv2.waitKey(5):
        #     pass
