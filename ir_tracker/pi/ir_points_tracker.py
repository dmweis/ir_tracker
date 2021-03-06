import cv2
import numpy as np

from ir_tracker.utils import debug_server, multicast, picam_wrapper, utility


def main():
    debug = True
    if debug:
        debug_image_container = debug_server.create_image_server()
    broadcaster = multicast.Broadcaster()
    counter = utility.FramerateCounter()
    use_otsu_thresholding = False
    binarization_threshold = 180

    capture_timer = utility.FramerateCounter()
    conv_thresh_timer = utility.FramerateCounter()
    contour_timer = utility.FramerateCounter()
    with picam_wrapper.picamera_opencv_video(resolution=(640, 480),
                                             framerate=30) as video_stream:
        for frame in video_stream:
            capture_timer.reset()
            debug_image_container["last_image"] = frame.copy()
            image = frame.copy()
            print("frame capture took", capture_timer.measure())

            conv_thresh_timer.reset()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            max_val = int(np.amax(gray))
            binarization_threshold = max_val - 50
            thresh, threshold_value = utility.threshold_image(
                gray, binarization_threshold, use_otsu_thresholding)
            print("conversion and thresholding took",
                  conv_thresh_timer.measure())

            contour_timer.reset()
            cnts_individual = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            # this is some weirdness where the return can be a tuple of 3 or 4 elements
            cnts_individual = cnts_individual[0] if len(
                cnts_individual) == 2 else cnts_individual[1]

            print("findContours took", contour_timer.measure())

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
                    if debug:
                        cv2.circle(image, (cX, cY), 1, (320, 159, 22), 4)
                except ZeroDivisionError:
                    pass

            if debug:
                colored_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                combined = utility.concat_images([colored_thresh, image])
                debug_image_container["combined"] = combined
            height, width, channels = image.shape
            data = {
                "frame_time": counter.measure(),
                "point_count": point_count,
                "height": height,
                "width": width,
                "channels": channels,
                "useing_otsu_thresholding": use_otsu_thresholding,
                "binarization_threshold": binarization_threshold,
                "points": point_centers,
            }
            broadcaster.send_json(data)


if __name__ == "__main__":
    main()
