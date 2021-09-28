from ir_tracker.utils import utility, picam_wrapper

with picam_wrapper.picamera_opencv_video(resolution=(640, 480),
                                         framerate=32) as video_stream:
    frame_timer = utility.FramerateCounter()
    for frame in video_stream:
        print("Frame time", frame_timer.measure())
