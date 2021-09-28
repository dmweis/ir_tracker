import contextlib
import numpy as np
import picamera

RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720
FRAMERATE = 30
RESOLUTION = (RESOLUTION_WIDTH, RESOLUTION_HEIGHT)


class CameraWrapper:
    def __init__(self, camera):
        self.camera = camera

    def get_frame(self):
        image = np.empty((RESOLUTION_HEIGHT * RESOLUTION_WIDTH * 3, ),
                         dtype=np.uint8)
        self.camera.capture(image, 'bgr')
        image = image.reshape((RESOLUTION_HEIGHT, RESOLUTION_WIDTH, 3))
        return image


@contextlib.contextmanager
def opencv_picamera():
    with picamera.PiCamera() as camera:
        camera.resolution = RESOLUTION
        camera.framerate = FRAMERATE
        yield CameraWrapper(camera)
