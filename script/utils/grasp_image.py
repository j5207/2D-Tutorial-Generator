import cv2
import numpy as np
from constant import Green_low, Green_high
camera = cv2.VideoCapture(0)
def warp(img):
        #pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
        pts1 = np.float32([[101,160],[531,133],[0,480],[640,480]])
        pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(640,480))
        return dst
while True:
    return_value,image = camera.read()
    image = warp(image)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    object_mask = 255 - cv2.inRange(hsv, Green_low, Green_high)
    cv2.imshow('image',object_mask)
    if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite('test.jpg',object_mask)
        cv2.imwrite('test_.jpg',image)
        break
camera.release()
cv2.destroyAllWindows()
