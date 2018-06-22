import numpy as np
import cv2
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((7 ,7), "uint8")
kernel1 = np.ones((3 ,3), "uint8")
while(1):
    ret, origin = cap.read()
    img = origin[120:360,160:480, :]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.rectangle(origin, (160, 120), (480, 360), (0, 255, 0), 1)
    mog_mask = fgbg.apply(img)
    mog_mask[mog_mask == 127] = 0
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv,np.array([119,59,37]),np.array([164,255,255]))
    hsv_mask = cv2.dilate(hsv_mask, kernel)
    # hsv_mask = cv2.GaussianBlur(hsv_mask,(3,3),-1)
    hsv_mask[hsv_mask > 0] = 255
    mask = cv2.subtract(mog_mask, hsv_mask)
    res=cv2.bitwise_and(img, img, mask = mask)

    (_,contours,hierarchy)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        contour = contours[max_index]  
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(img,[box],0,(0,0,255),2)
    # res = cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel1)
    # res_hsv = cv2.cvtColor(res,cv2.COLOR_BGR2HSV)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()