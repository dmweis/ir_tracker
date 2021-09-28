from ir_tracker import debug_server
import picamera

RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720
FRAMERATE = 30
RESOLUTION = (RESOLUTION_WIDTH, RESOLUTION_HEIGHT)

if __name__ == "__main__":
    debug_image_container = debug_server.create_image_server()

    with picamera.PiCamera() as camera:
        camera.resolution = RESOLUTION
        camera.framerate = FRAMERATE
        while True:
            image = np.empty((RESOLUTION_HEIGHT * RESOLUTION_WIDTH * 3, ),
                             dtype=np.uint8)
            camera.capture(image, 'bgr')
            image = image.reshape((RESOLUTION_HEIGHT, RESOLUTION_WIDTH, 3))
            debug_image_container["last_image"] = image
