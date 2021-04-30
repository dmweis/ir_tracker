import time
import picamera
import numpy as np
import cv2

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

    for i in range(10):
        image = np.empty((RESOLUTION_HEIGHT * RESOLUTION_WIDTH * 3, ),
                         dtype=np.uint8)
        camera.capture(image, 'bgr')
        image = image.reshape((RESOLUTION_HEIGHT, RESOLUTION_WIDTH, 3))
        file_name = f'image_{i}.png'
        cv2.imwrite(file_name, image)
        print(f"Saved file {file_name}")
        time.sleep(2)
