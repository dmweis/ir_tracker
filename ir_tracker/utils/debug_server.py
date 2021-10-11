import threading

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, abort

app = Flask(__name__)

image_container = {}


def camera_stream(image_name):
    while True:
        _, payload = cv2.imencode('.jpg', image_container[image_name])
        frame = payload.tobytes()
        yield (b'--frame/r/n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


INDEX_TEMPLATE = """
<body>
    <ul>
    {% for image_stream in image_stream_list %}
        <li>
            <a href="{{url_for('stream_page', image_name=image_stream)}}">{{image_stream}}</a>
        </li>
    {% endfor %}
    </ul>
</body>
"""

STREAM_PAGE_TEMPLATE = """
<body>
    <img src="{{url_for('video_feed', image_name=image_name)}}" height="100%">
</body>
"""


@app.route('/image_stream/<image_name>')
def video_feed(image_name):
    return Response(camera_stream(image_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/image_frame/<image_name>')
def video_frame(image_name):
    image = image_container.get(image_name, None)
    if image is None:
        return abort(404)
    _, payload = cv2.imencode('.jpg', image)
    frame = payload.tobytes()
    return Response(frame, mimetype='image/jpeg')


@app.route("/stream_page/<image_name>")
def stream_page(image_name):
    return render_template_string(STREAM_PAGE_TEMPLATE, image_name=image_name)


@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE,
                                  image_stream_list=image_container.keys())


def create_image_server():
    """
    Create flask image debug server.

    Warning! This is a very hacky flask server. Make sure to only run this once as it uses global stare

    Returns:
        Dictionary into which you can insert images with string keys
    """
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8000)).start()
    return image_container
