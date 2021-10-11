import cv2
from ir_tracker.utils import debug_server, multicast, picam_wrapper, utility


def main():
    orb = cv2.ORB_create()
    debug_image_container = debug_server.create_image_server()
    counter = utility.FramerateCounter()

    with picam_wrapper.picamera_opencv_video(resolution=(640, 480),
                                             framerate=30) as video_stream:
        for frame in video_stream:
            debug_image_container["last_image"] = frame.copy()
            key_points, descriptors = orb.detectAndCompute(frame, None)
            orb_image = cv2.drawKeypoints(frame,
                                          key_points,
                                          outImage=None,
                                          flags=2,
                                          color=(0, 255, 255))
            debug_image_container["orb_image"] = orb_image
            print("FPS: ", counter.measure_fps())


if __name__ == "__main__":
    main()
