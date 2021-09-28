import cv2
import numpy as np

from ir_tracker import debug_server, multicast, picam_wrapper, utility


def main():
    debug_image_container = debug_server.create_image_server()
    broadcaster = multicast.Broadcaster()
    counter = utility.FramerateCounter()
    use_otsu_thresholding = False
    binarization_threshold = 180
    with picam_wrapper.opencv_picamera() as camera:
        while True:
            image = camera.get_frame()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh, threshold_value = utility.threshold_image(
                gray, binarization_threshold, use_otsu_thresholding)

            cnts_individual = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            # this is some weirdness where the return can be a tuple of 3 or 4 elements
            cnts_individual = cnts_individual[0] if len(
                cnts_individual) == 2 else cnts_individual[1]
            point_count = len(cnts_individual)

            point_centers = []

            for c in cnts_individual:
                # Get bounding rect
                x, y, w, h = cv2.boundingRect(c)

                # Find centroid
                M = cv2.moments(c)
                try:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    point_centers.append((cX, cY))

                    # Draw the contour and center of the shape on the image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0),
                                  2)
                    cv2.circle(image, (cX, cY), 1, (320, 159, 22), 8)
                    cv2.putText(image, '({}, {})'.format(cX, cY), (x, y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100),
                                2)
                except ZeroDivisionError:
                    pass

            colored_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            combined = utility.concat_images([colored_thresh, image])

            data = {
                "points": point_centers,
                "frame_time": counter.measure(),
                "point_count": point_count,
                "useing_otsu_thresholding": use_otsu_thresholding,
                "binarization_threshold": binarization_threshold,
            }

            broadcaster.send_json(data)

            debug_image_container["combined"] = combined


if __name__ == "__main__":
    main()