import time
from pathlib import Path

import cv2
from ir_tracker.utils import calibration_manager, debug_server, picam_wrapper


def draw_info(image, text):
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2, cv2.LINE_AA)


CHESSBOARD_HEIGHT = 8
CHESSBOARD_WIDTH = 5
PICTURE_TIME = 3
NUMBER_OF_IMAGES = 10


def main():
    debug_image_container = debug_server.create_image_server()

    with picam_wrapper.picamera_opencv_video(resolution=(640, 480),
                                             framerate=30) as video_stream:
        calibration_images = []
        for frame in video_stream:
            while len(calibration_images) < 10:
                start_time = time.time()
                while True:
                    current_time = time.time()
                    time_delta = current_time - start_time
                    if time_delta > PICTURE_TIME:
                        clean_image = frame.copy()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        found_chessboard, corners = cv2.findChessboardCorners(
                            gray, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), None)
                        cv2.drawChessboardCorners(
                            frame, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH),
                            corners, found_chessboard)
                        if found_chessboard:
                            calibration_images.append(clean_image)
                            draw_info(frame, "Image saved")
                            debug_image_container["calib"] = frame
                            # time.sleep(1)
                        else:
                            draw_info(frame, "Chessboard not found")
                            debug_image_container["calib"] = frame
                            # time.sleep(1)
                        break
                    draw_info(
                        frame,
                        f"{PICTURE_TIME - time_delta:.1f}s left, {len(calibration_images)}/{NUMBER_OF_IMAGES}"
                    )
                    # detect chessboard
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    found_chessboard, corners = cv2.findChessboardCorners(
                        gray, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), None)
                    cv2.drawChessboardCorners(
                        frame, (CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH), corners,
                        found_chessboard)
                    debug_image_container["calib"] = frame

        image_directory = Path.home().joinpath("calibration_images")
        image_directory.mkdir(parents=True, exist_ok=True)
        print(f"Saving images to {image_directory}")
        for i, image in enumerate(calibration_images):
            cv2.imwrite(f"{str(image_directory)}/image_{i}.png", image)
        print("images saved")

        print("Calibrating")
        calibartion = calibration_manager.calibarate_from_images(
            calibration_images, CHESSBOARD_HEIGHT, CHESSBOARD_WIDTH, 500)

        calibration_dir = Path.home().joinpath("calibration")
        calibration_dir.mkdir(parents=True, exist_ok=True)
        calibration_path = calibration_dir.joinpath("picamera_calibration.yml")
        print(f"Saving calibration to {calibration_path}")
        calibartion.save_yaml(str(calibration_path))
        calibartion_read = calibration_manager.ImageCalibration.load_yaml(
            calibration_path)

        for image in calibration_images:
            undisorted = calibartion_read.undistort_image(image, False)
            combined = cv2.vconcat((image, undisorted))
            debug_image_container["calib"] = combined


if __name__ == "__main__":
    main()