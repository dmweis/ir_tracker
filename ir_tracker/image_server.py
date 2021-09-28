import threading

import cv2
import numpy as np
import picamera
from flask import Flask, Response, render_template_string

RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720
FRAMERATE = 30
RESOLUTION = (RESOLUTION_WIDTH, RESOLUTION_HEIGHT)

app = Flask(__name__)

last_image = None


def camera_stream():
    while True:
        _, payload = cv2.imencode('.jpg', last_image)
        frame = payload.tobytes()
        yield (b'--frame/r/n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


INDEX_TEMPLATE = """
<body>
    <img src="{{ url_for('image_stream') }}" width="100%">
</body>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE)


@app.route('/image_stream')
def video_feed():
    return Response(camera_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8000)).start()

    with picamera.PiCamera() as camera:
        camera.resolution = RESOLUTION
        camera.framerate = FRAMERATE
        while True:
            image = np.empty((RESOLUTION_HEIGHT * RESOLUTION_WIDTH * 3, ),
                             dtype=np.uint8)
            camera.capture(image, 'bgr')
            image = image.reshape((RESOLUTION_HEIGHT, RESOLUTION_WIDTH, 3))
            last_image = image
