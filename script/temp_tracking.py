from __future__ import print_function
import cv2
import numpy as np
from constant import *


def warp_img(img):
    #pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
    pts1 = np.float32([[101,160],[531,133],[0,480],[640,480]])
    pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(640,480))
    return dst

def camrectify(frame):
        mtx = np.array([
            [509.428319, 0, 316.944024],
            [0.000000, 508.141786, 251.243128],
            [0.000000, 0.000000, 1.000000]
        ])
        dist = np.array([
            0.052897, -0.155430, 0.005959, 0.002077, 0.000000
        ])
        return cv2.undistort(frame, mtx, dist)

def get_crit(mask):
    (_,contours, hierarchy)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    crit = None
    for i , contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area and hierarchy[0, i, 3] == -1:
            max_area = area
            crit = area
    return crit

def get_objectmask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, Green_low, Green_high)
    hand_mask = cv2.inRange(hsv, Hand_low, Hand_high)
    hand_mask = cv2.dilate(hand_mask, kernel = np.ones((7,7),np.uint8))
    skin_mask = cv2.inRange(hsv, Skin_low, Skin_high)
    skin_mask = cv2.dilate(skin_mask, kernel = np.ones((7,7),np.uint8))
    thresh = 255 - green_mask
    thresh = cv2.subtract(thresh, hand_mask)
    thresh = cv2.subtract(thresh, skin_mask)
    thresh[477:, 50:610] = 0
    return thresh

class temp_tracking():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.boxls = []
        self.area_memory = {}
        OK, origin = self.cap.read()
        if OK:
            rect = camrectify(origin)
            warp = warp_img(rect)
            thresh = get_objectmask(warp)
            draw_img1 = warp.copy()
            self.get_bound(draw_img1, thresh, visualization=True)
            for i, (x, y, w, h) in enumerate(self.boxls):
                self.area_memory[i] = get_crit(thresh[y:y+h, x:x+w])

    def update(self):
        self.boxls = []
        OK, origin = self.cap.read()
        if OK:
            rect = camrectify(origin)
            warp = warp_img(rect)
            thresh = get_objectmask(warp)
            draw_img1 = warp.copy()
            self.get_bound(draw_img1, thresh, visualization=True)
            for i, (x, y, w, h) in enumerate(self.boxls):
                Max_diff = np.inf
                index = None
                for ind, area in self.area_memory.iteritems():
                    det_area = get_crit(thresh[y:y+h, x:x+w])
                    diff = abs(det_area - area)
                    if diff < Max_diff:
                        Max_diff = diff
                        index = ind
                cv2.rectangle(draw_img1,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(draw_img1,str(index),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
            cv2.imshow('img', draw_img1)
            #print(self.area_memory)


    def get_bound(self, img, object_mask, visualization=True):
        (_,object_contours, object_hierarchy)=cv2.findContours(object_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(object_contours) > 0:
            for i , contour in enumerate(object_contours):
                area = cv2.contourArea(contour)
                if area>100 and area < 100000 and object_hierarchy[0, i, 3] == -1:					
                    M = cv2.moments(contour)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    x,y,w,h = cv2.boundingRect(contour)
                    self.boxls.append((x, y, w, h))
        if len(self.boxls) > 0:
            boxls_arr = np.array(self.boxls)
            self.boxls = boxls_arr[boxls_arr[:, 0].argsort()].tolist()
        # for i in range(len(self.boxls)): 
        #     # if visualization: 
        #     #     ind = max(range(len(self.boxls)), key=lambda i:self.boxls[i][2]*self.boxls[i][3])
        #     x,y,w,h = self.boxls[i]
        #     cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #     cv2.putText(img,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
        # cv2.imshow('show', img)
    

    def __del__(self):
        self.cap.release()

if __name__ == '__main__':
    temp = temp_tracking()
    while True:
        temp.update()
        k = cv2.waitKey(1) & 0xFF # large wait time to remove freezing
        if k == 113 or k == 27:
            break
    cv2.destroyAllWindows()