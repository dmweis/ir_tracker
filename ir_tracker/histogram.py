import cv2
import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt


def read_image():
    resp = urlopen('http://camerapi.local:8080/?action=snapshot')
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


plt.ion()
plt.show()

if __name__ == "__main__":
    while True:
        image = read_image()
        image = cv2.resize(image, (640, 480))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # threshold
        hist = (hist < 50) * hist
        # plt.hist(gray.ravel(), 256, [0, 256])
        plt.clf()
        plt.plot(hist)
        plt.draw()

        cv2.imshow("thresh", gray)
        ret = cv2.waitKey(100)
        if 27 == ret:
            # 27 is esc
            break
