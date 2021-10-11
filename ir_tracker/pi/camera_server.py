from ir_tracker.utils import debug_server, picam_wrapper

if __name__ == "__main__":
    debug_image_container = debug_server.create_image_server()

    with picam_wrapper.opencv_picamera() as camera:
        while True:
            debug_image_container["last_image"] = camera.get_frame()
