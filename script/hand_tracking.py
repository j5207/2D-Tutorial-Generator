import cv2
import numpy as np

class hand_tracking():
    def __init__(self, cap):
        _, frame = cap.read()
        blur = cv2.blur(frame,(3,3))
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv,np.array([0,101,0]),np.array([10,255,255]))  
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)

        median = cv2.medianBlur(dilation2,5)
        _,thresh = cv2.threshold(median,127,255,0)
        cv2.imshow('thresh', thresh)
        

        _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
        

        max_area=1000
        try:	
            for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    cnts = contours[i]
                    epsilon = 0.001*cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,epsilon,True)
                    hull = cv2.convexHull(cnts)
                    #Find convex defects
                    hull2 = cv2.convexHull(cnts,returnPoints = False)
                    defects = cv2.convexityDefects(cnts,hull2)
                    cv2.drawContours(frame,[approx],-1,(0, 255, 0),2)
        except:
            pass
                
        cv2.imshow('hand_tracking',frame)

# if __name__ == '__main__':
#     cap = cv2.VideoCapture(0)
#     hand_tracking(cap)
#     cap.release()
#     cv2.destroyAllWindows()