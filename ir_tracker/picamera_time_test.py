import time
import picamera
import numpy as np
import cv2

from ir_tracker.utils import utility

RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720
FRAMERATE = 30
RESOLUTION = (RESOLUTION_WIDTH, RESOLUTION_HEIGHT)

with picamera.PiCamera() as camera:
    camera.resolution = RESOLUTION
    camera.framerate = FRAMERATE
    camera.iso = 100
    time.sleep(2)
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g

    frame_timer = utility.FramerateCounter()
    while True:
        image = np.empty((RESOLUTION_HEIGHT * RESOLUTION_WIDTH * 3, ),
                         dtype=np.uint8)
        camera.capture(image, 'bgr')
        image = image.reshape((RESOLUTION_HEIGHT, RESOLUTION_WIDTH, 3))
        print("Frame time", frame_timer.measure())