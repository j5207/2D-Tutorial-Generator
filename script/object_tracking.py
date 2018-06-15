#importing modules

import cv2
import numpy as np

class object_tracking():
    def __init__(self, cap):
		_, img = cap.read()

		hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

		dic = {
			'red' : [np.array([136,87,111],np.uint8), np.array([180,255,255],np.uint8), 500, (0,0,255)],
			'blue' : [np.array([113,50,50],np.uint8), np.array([130,255,255],np.uint8), 800, (255,0,0)],
			'yello' : [np.array([22,60,200],np.uint8), np.array([60,255,255],np.uint8), 500, (0,255,0)],
			'green' : [np.array([30,19,76],np.uint8), np.array([88,128,162],np.uint8), 500, (0,255,0)],
		}

		circle_dic = {
			'washer' : [np.array([0,0,0],np.uint8), np.array([22,255,89],np.uint8), 100, (120,120,0)] ,
			'connector' : [np.array([136,87,111],np.uint8), np.array([180,255,255],np.uint8), 500, (0,0,255)]
		}


		kernal = np.ones((3 ,3), "uint8")
		kernel_square = np.ones((11,11),np.uint8)
		kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

		for k in circle_dic:
			mask = cv2.inRange(hsv, circle_dic[k][0], circle_dic[k][1])
			mask = cv2.dilate(mask, kernal)
			# mask = cv2.erode(mask,kernal)		
			mask = cv2.GaussianBlur(mask,(11,11),-1)
			# mask = cv2.Laplacian(mask,cv2.CV_64F)
			# mask = cv2.bilateralFilter(mask,9,75,75)
			# mask = cv2.medianBlur(mask,5)

			# cv2.imshow('circle', mask)
			# cv2.imwrite('circle.jpg', mask)
			try:
				circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
								param1=200,param2=25,minRadius=0,maxRadius=70)		
				circles = np.uint16(np.around(circles))
			
				print circles
				for i in np.squeeze(circles):
						if i[2] > 5:
							# draw the outer circle
						# draw the outer circle
							cv2.circle(img,(i[0],i[1]),i[2],circle_dic[k][3],2)
							# draw the center of the circle
							cv2.circle(img,(i[0],i[1]),3,circle_dic[k][3],3)
							cv2.putText(img,k,(i[0],i[1]),cv2.FONT_HERSHEY_SIMPLEX, 1.0, circle_dic[k][3])		
							# draw the center of the circle
							cv2.circle(img,(i[0],i[1]),3,circle_dic[k][3],3)
							cv2.putText(img,k,(i[0],i[1]),cv2.FONT_HERSHEY_SIMPLEX, 1.0, circle_dic[k][3])
			except:
				print 'no circles'

				

		for k in dic:
			mask = cv2.inRange(hsv, dic[k][0], dic[k][1])
			dilation = cv2.dilate(mask,kernel_ellipse,iterations = 1)
			erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
			dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
			filtered = cv2.medianBlur(dilation2,5)
			kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
			dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
			kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
			median = cv2.medianBlur(dilation2,5)
			ret,mask = cv2.threshold(median,127,255,0)
			# mask = cv2.dilate(mask, kernal)
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
		cv2.imshow("Color Tracking",img)




