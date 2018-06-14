#importing modules

import cv2
import numpy as np

#capturing video through webcam
cap=cv2.VideoCapture(0)

while(1):
	_, img = cap.read()

	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

	dic = {
		'red' : [np.array([136,87,111],np.uint8), np.array([180,255,255],np.uint8), 500, (0,0,255)],
		'blue' : [np.array([113,50,50],np.uint8), np.array([130,255,255],np.uint8), 800, (255,0,0)],
		'yello' : [np.array([22,60,200],np.uint8), np.array([60,255,255],np.uint8), 500, (0,255,0)],
		'green' : [np.array([30,19,76],np.uint8), np.array([88,128,162],np.uint8), 500, (0,255,0)],
		'washer' : [np.array([0,40,80],np.uint8), np.array([6,112,129],np.uint8), 300, (120,120,0)],
	}

	hand_dic = {
		'hand' : [np.array([0,56,100],np.uint8), np.array([17,255,255],np.uint8), 500, (0,0,255)],
	}

	kernal = np.ones((5 ,5), "uint8")

	for k in dic:
		mask = cv2.inRange(hsv, dic[k][0], dic[k][1])
		mask = cv2.dilate(mask, kernal)
		(_,contours,hierarchy)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if(area>dic[k][2]):
				M = cv2.moments(contour)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				cv2.circle(img, (cx, cy), 10, dic[k][3], 3)
				x,y,_,_ = cv2.boundingRect(contour)
				rect = cv2.minAreaRect(contour)
				box = np.int0(cv2.boxPoints(rect))
				cv2.drawContours(img,[box],0,dic[k][3],2)
				cv2.putText(img,k,(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, dic[k][3])

	#cv2.imshow("Redcolour",red)
	cv2.imshow("Color Tracking",img)
	#cv2.imshow("red",res)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break



