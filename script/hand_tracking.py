import cv2
import numpy as np
import heapq


class hand_tracking():
    def __init__(self, image):
        frame = image.copy()
        self.radius_thresh = 0.05

        #_, frame = cap.read()
        #frame = self.warp(frame)
        blur = cv2.blur(frame,(3,3))
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        kernal = np.ones((7 ,7), "uint8")
        mask = cv2.inRange(hsv,np.array([128,38,0]),np.array([139,255,255]))
        mask = cv2.dilate(mask, kernal)
        mask = cv2.GaussianBlur(mask,(1,1),-1)  
        # kernel_square = np.ones((11,11),np.uint8)
        # kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

        # dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        # erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
        # dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
        # filtered = cv2.medianBlur(dilation2,5)
        # kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        # dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)

        # median = cv2.medianBlur(dilation2,5)
        # _,thresh = cv2.threshold(median,127,255,0)
        # cv2.imshow('thresh', thresh)
        

        _, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        self.hand_cnt = [] 
        self.only_point = None
        self.center = None
        try:	
            for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                # if(area>max_area):
                cnts = contours[i]
                
                # M = cv2.moments(cnts)
                # cx = int(M['m10']/M['m00'])
                # cy = int(M['m01']/M['m00'])
                # cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                epsilon = 0.001*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                hull = cv2.convexHull(cnts)
                frame,hand_center,hand_radius = self.mark_hand_center(frame, cnts)
                
                frame,finger=self.mark_fingers(frame,hull,hand_center,hand_radius)
                # #Find convex defects
                # hull2 = cv2.convexHull(cnts,returnPoints = False)
                # defects = cv2.convexityDefects(cnts,hull2)
                cv2.drawContours(frame,[approx],-1,(0, 255, 0),1)
                self.hand_cnt.append([approx])
        except Exception as e:
            print e
                
        cv2.imshow('hand_tracking',frame) 
    def get_result(self):
        # return self.hand_cnt
        return self.only_point, self.center

    def mark_hand_center(self, frame_in,cont):    
        max_d=0
        pt=(0,0)
        x,y,w,h = cv2.boundingRect(cont)
        for ind_y in xrange(int(y),int(y+h)): #around 0.25 to 0.6 region of height (Faster calculation with ok results)
            for ind_x in xrange(int(x),int(x+w)): #around 0.3 to 0.6 region of width (Faster calculation with ok results)
                dist= cv2.pointPolygonTest(cont,(ind_x,ind_y),True)
                if(dist>max_d):
                    max_d=dist
                    pt=(ind_x,ind_y)
        cv2.circle(frame_in,pt,int(max_d),(255,0,0),2)
        return frame_in,pt,max_d

    def mark_fingers(self, frame_in,hull,pt,radius):
      
        finger=[(hull[0][0][0],hull[0][0][1])]
        j=0

        cx = pt[0]
        cy = pt[1]
        self.center = (cx, cy)
        for i in range(len(hull)):
            dist = np.sqrt((hull[-i][0][0] - hull[-i+1][0][0])**2 + (hull[-i][0][1] - hull[-i+1][0][1])**2)
            if (dist>15):
                if(j==0):
                    finger=[(hull[-i][0][0],hull[-i][0][1])]
                else:
                    finger.append((hull[-i][0][0],hull[-i][0][1]))
                j=j+1

        finger = filter(lambda x: x[0] < cx, finger)
        dis_center_ls = []        
        for i in range(len(finger)):
            dist = np.sqrt((finger[i][0]- cx)**2 + (finger[i][1] - cy)**2)
            dis_center_ls.append(dist)
        if len(dis_center_ls) >= 2:
            largest_two = heapq.nlargest(2, dis_center_ls)
            
            if largest_two[0] > largest_two[1] * 1.3 and largest_two[0]>80 and radius>26 and largest_two[0] - largest_two[1] > 30:
                #print largest_two[0] , largest_two[1]
                cv2.putText(frame_in,"pointing",(int(0.38*frame_in.shape[1]),int(0.12*frame_in.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
                self.only_point = finger[dis_center_ls.index(largest_two[0])]
        elif len(dis_center_ls) == 1:
            if dis_center_ls[0] > 100:
                #print "only, {}".format(dis_center_ls[0])
                cv2.putText(frame_in,"pointing",(int(0.38*frame_in.shape[1]),int(0.12*frame_in.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
                self.only_point = finger[0]

            

        for k in range(len(finger)):
            cv2.circle(frame_in,finger[k],10,255,2)
            cv2.line(frame_in,finger[k],(cx,cy),255,2)
        return frame_in,finger
    
    def warp(self, img):
        #pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
        pts1 = np.float32([[101,160],[531,133],[0,480],[640,480]])
        pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(640,480))
        return dst
    
        



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        hand_tracking(cap)
        k = cv2.waitKey(1) & 0xFF # large wait time to remove freezing
        if k == 113 or k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()