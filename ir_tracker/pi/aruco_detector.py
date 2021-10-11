import cv2
from ir_tracker.utils import debug_server, multicast, picam_wrapper, utility


def main():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters_create()
    debug_image_container = debug_server.create_image_server()
    counter = utility.FramerateCounter()

    with picam_wrapper.picamera_opencv_video(resolution=(640, 480),
                                             framerate=30) as video_stream:
        for frame in video_stream:
            debug_image_container["last_image"] = frame.copy()
            (corners, ids,
             rejected) = cv2.aruco.detectMarkers(frame,
                                                 aruco_dict,
                                                 parameters=aruco_params)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            debug_image_container["aruco_image"] = frame
            print("loop took", counter.measure())


if __name__ == "__main__":
    main()
