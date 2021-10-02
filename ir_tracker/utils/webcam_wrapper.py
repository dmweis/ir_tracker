import cv2
import contextlib

FRAMERATE = 30
DEFAULT_ID = 0


def video_stream_iterator(video_stream):
    while True:
        _, frame = video_stream.read()
        yield frame


@contextlib.contextmanager
def webcam(id=DEFAULT_ID, framerate=FRAMERATE):
    try:
        video = cv2.VideoCapture(id)
        video.set(cv2.CAP_PROP_FPS, framerate)
        yield video_stream_iterator(video)
    finally:
        video.release()
