#importing modules

import cv2   
import numpy as np

#capturing video through webcam
cap=cv2.VideoCapture(0)

while(1):
	_, img = cap.read()
		
	#converting frame(img i.e BGR) to HSV (hue-saturation-value)

	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


	#definig the range of red color
	red_lower=np.array([136,87,111],np.uint8)
	red_upper=np.array([180,255,255],np.uint8)

	#defining the Range of Blue color
	blue_lower=np.array([113,50,50],np.uint8)
	blue_upper=np.array([130,255,255],np.uint8)
	
	#defining the Range of yellow color
	yellow_lower=np.array([22,60,200],np.uint8)
	yellow_upper=np.array([60,255,255],np.uint8)

	#defining the Range of green color
	green_lower=np.array([30,19,76],np.uint8)
	green_upper=np.array([88,128,162],np.uint8)

	#defining the Range of washer color
	washer_lower=np.array([0,40,80],np.uint8)
	washer_upper=np.array([6,112,129],np.uint8)

	#finding the range of red,blue and yellow color in the image
	red=cv2.inRange(hsv, red_lower, red_upper)
	blue=cv2.inRange(hsv,blue_lower,blue_upper)
	yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)
	washer=cv2.inRange(hsv,washer_lower,washer_upper)
	green=cv2.inRange(hsv,green_lower,green_upper)
	
	#Morphological transformation, Dilation  	
	kernal = np.ones((5 ,5), "uint8")

	red=cv2.dilate(red, kernal)
	res=cv2.bitwise_and(img, img, mask = red)

	blue=cv2.dilate(blue,kernal)

	yellow=cv2.dilate(yellow,kernal)

	washer=cv2.dilate(washer,kernal)

	green=cv2.dilate(green,kernal)  


	#Tracking the Red Color
	(_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>500):
			M = cv2.moments(contour)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			cv2.circle(img, (cx, cy), 10, (0, 0, 255), 3)
			x,y,_,_ = cv2.boundingRect(contour)
			rect = cv2.minAreaRect(contour)
			box = np.int0(cv2.boxPoints(rect))
			cv2.drawContours(img,[box],0,(0,0,255),2)
			cv2.putText(img,"pipe1",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
			
	#Tracking the Blue Color
	(_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>800):
			M = cv2.moments(contour)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			cv2.circle(img, (cx, cy), 10, (255, 0, 0), 3)
			x,y,_,_ = cv2.boundingRect(contour)
			rect = cv2.minAreaRect(contour)
			box = np.int0(cv2.boxPoints(rect))
			cv2.drawContours(img,[box],0,(255,0,0),2)
			cv2.putText(img,"pipe2",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
	
	#Tracking the washer color
	(_,contours,hierarchy)=cv2.findContours(washer,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>200):
			M = cv2.moments(contour)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			cv2.circle(img, (cx, cy), 10, (120, 120, 0), 3)
			x,y,_,_ = cv2.boundingRect(contour)
			rect = cv2.minAreaRect(contour)
			box = np.int0(cv2.boxPoints(rect))
			cv2.drawContours(img,[box],0,(120,120,0),2)
			cv2.putText(img,"washer",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120,120,0))

	#Tracking the yellow Color
	(_,contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>500):
			M = cv2.moments(contour)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			cv2.circle(img, (cx, cy), 10, (0, 255, 0), 3)
			x,y,_,_ = cv2.boundingRect(contour)
			rect = cv2.minAreaRect(contour)
			box = np.int0(cv2.boxPoints(rect))
			cv2.drawContours(img,[box],0,(0,255,0),2)
			cv2.putText(img,"yellow",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

	#Tracking the green Color
	(_,contours,hierarchy)=cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area>500):
			M = cv2.moments(contour)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			cv2.circle(img, (cx, cy), 10, (0, 255, 0), 3)
			x,y,_,_ = cv2.boundingRect(contour)
			rect = cv2.minAreaRect(contour)
			box = np.int0(cv2.boxPoints(rect))
			cv2.drawContours(img,[box],0,(0,255,0),2)
			cv2.putText(img,"pipe3",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))  
			
		   
	#cv2.imshow("Redcolour",red)
	#cv2.imshow("Color Tracking",img)
	cv2.imshow("red",res) 	
	if cv2.waitKey(10) & 0xFF == ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break  
