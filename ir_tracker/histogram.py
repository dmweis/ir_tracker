import cv2
import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt
from ir_tracker.utils import utility

plt.ion()
plt.show()

if __name__ == "__main__":
    while True:
        image = utility.request_image()
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
