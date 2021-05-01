import cv2
import numpy as np
from urllib.request import urlopen


def read_image():
    resp = urlopen('http://camerapi.local:8080/?action=snapshot')
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    width, height = image.shape[:2]
    return image


if __name__ == "__main__":
    # for image_id in range(10):
    use_otsu_thresholding = False
    binarization_threshold = 180
    while True:
        # image = cv2.imread(f"calibration_images/image000{image_id}.jpg")
        image = read_image()
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
        combined = cv2.vconcat((original, image))
        cv2.imshow("thresh", combined)
        cv2.waitKey(50)
        # while 113 != cv2.waitKey(5):
        #     pass
