import cv2
from ir_tracker.utils import webcam_wrapper, utility

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
aruco_params = cv2.aruco.DetectorParameters_create()

while True:
    frame = utility.request_image(
        url="http://camerapi2.local:8000/image_frame/last_image")
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
                                                       aruco_dict,
                                                       parameters=aruco_params)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
