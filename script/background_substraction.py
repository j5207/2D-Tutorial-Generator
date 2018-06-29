import numpy as np
import cv2
import time

toarr = lambda x, y, z : np.array([x, y, z], np.uint8)
mean_ = lambda x : np.sum(x) // np.count_nonzero(x)

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((7 ,7), "uint8")
kernel1 = np.ones((3 ,3), "uint8")
tracker = cv2.TrackerMIL_create()
sift = cv2.xfeatures2d.SIFT_create()
while(1):
    ret, origin = cap.read()
    # set ROI
    img = origin[120:360,160:480, :]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.rectangle(origin, (160, 120), (480, 360), (0, 255, 0), 1)
    # background segmentation
    mog_mask = fgbg.apply(img)
    # get rid of shadow
    mog_mask[mog_mask == 127] = 0
    # subtract glove
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv,np.array([119,59,37]),np.array([164,255,255]))
    hsv_mask = cv2.dilate(hsv_mask, kernel)
    cv2.imshow('ggg',hsv_mask)
    hsv_mask[hsv_mask > 0] = 255
    mask = cv2.subtract(mog_mask, hsv_mask)
    res=cv2.bitwise_and(img, img, mask = mask)
    
    


    # hsv_res=cv2.cvtColor(res,cv2.COLOR_BGR2HSV)
    # hsv_res = cv2.erode(hsv_res,kernel1,iterations = 1)
    # temp = hsv_res.copy()

    # h_low, h_high = mean_(hsv_res[:,:,0]) - 20, mean_(hsv_res[:,:,0]) + 20
    # s_low, s_high = mean_(hsv_res[:,:,1]), 255
    # v_low, v_high = mean_(hsv_res[:,:,2]), 255
    
   
    
    
    # print h_low, s_low, v_low
    # # h_low, h_high = np.min(hsv_res[:,:,0]), np.max(hsv_res[:,:,0])
    # # s_low, s_high = np.min(hsv_res[:,:,1]), np.max(hsv_res[:,:,1])
    # # v_low, v_high = np.min(hsv_res[:,:,2]), np.max(hsv_res[:,:,2])
    # # print np.unique(hsv_res)
    # mask = cv2.inRange(origin, toarr(h_low, s_low, v_low), toarr(h_high, s_high, v_high))



    # gray= cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    # kp, = sift.detectAndCompute(gray,None) 
    # cv2.drawKeypoints(img,kp, img)
    # draw bounding box
    (_,contours,hierarchy)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        contour = contours[max_index]  
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(img,[box],0,(0,0,255),2)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()