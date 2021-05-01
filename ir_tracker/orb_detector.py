import cv2
import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt


def read_image():
    resp = urlopen('http://camerapi.local:8080/?action=snapshot')
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def draw_info(image, text):
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2, cv2.LINE_AA)


while True:
    image = read_image()
    orb = cv2.ORB_create()
    key_points, descriptors = orb.detectAndCompute(image, None)
    orb_image = cv2.drawKeypoints(image, key_points, outImage=None, flags=2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 3, 255, -1)

    draw_info(orb_image, "orb")
    draw_info(image, "good features")

    combined = cv2.vconcat((orb_image, image))
    cv2.imshow("image", combined)
    ret = cv2.waitKey(100)
    if 27 == ret:
        # 27 is esc
        break