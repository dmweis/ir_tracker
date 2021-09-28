import time
import contextlib
import numpy as np
import picamera
from picamera.array import PiRGBArray

RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720
FRAMERATE = 30
RESOLUTION = (RESOLUTION_WIDTH, RESOLUTION_HEIGHT)


class CameraImageWrapper:
    def __init__(self, camera):
        self.camera = camera

    def get_frame(self):
        image = np.empty((RESOLUTION_HEIGHT * RESOLUTION_WIDTH * 3, ),
                         dtype=np.uint8)
        self.camera.capture(image, 'bgr')
        image = image.reshape((RESOLUTION_HEIGHT, RESOLUTION_WIDTH, 3))
        return image


@contextlib.contextmanager
def picamera_opencv_image():
    with picamera.PiCamera() as camera:
        camera.resolution = RESOLUTION
        camera.framerate = FRAMERATE
        yield CameraImageWrapper(camera)


def video_stream_iterator(camera, raw_capture):
    for frame in camera.capture_continuous(raw_capture,
                                           format="bgr",
                                           use_video_port=True):
        yield raw_capture.array
        raw_capture.truncate(0)


@contextlib.contextmanager
def picamera_opencv_video(resolution=RESOLUTION, framerate=FRAMERATE):
    with picamera.PiCamera() as camera:
        camera.resolution = resolution
        camera.framerate = framerate
        with picamera.array.PiRGBArray(camera, size=resolution) as raw_capture:
            time.sleep(0.5)
            yield video_stream_iterator(camera, raw_capture)
