from __future__ import print_function
import numpy as np
import cv2
import time

toarr = lambda x, y, z : np.array([x, y, z], np.uint8)
mean_ = lambda x : np.sum(x) // np.count_nonzero(x)

class object_detector():
	def __init__(self, cap):
		self.boxls = []

		_, origin = cap.read()
		#-------------warp the image---------------------#
		warp = self.warp(origin)
		#-------------segment the object----------------#
		hsv = cv2.cvtColor(warp,cv2.COLOR_BGR2HSV)        
		green_mask = cv2.inRange(hsv, np.array([57,145,0]), np.array([85,255,255]))
		res=cv2.bitwise_and(warp, warp, mask = green_mask)
		object_mask = cv2.subtract(warp, res)
		gray = cv2.cvtColor(object_mask,cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		draw_img = warp.copy()
		#-------------get the bounding box--------------
		self.get_minRect(draw_img, thresh, only=False)
		cv2.imshow('green_mask', draw_img)


	def get_minRect(self, img, mask, only=True, visualization=True):
		(_,contours,_)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) > 0:
			if only:
				areas = [cv2.contourArea(c) for c in contours]
				max_index = np.argmax(areas)
				contour = contours[max_index]
				self.boxls.append(contour)
				hull = cv2.convexHull(contour)
				rect = cv2.minAreaRect(contour)
				box = np.int0(cv2.boxPoints(rect))
				x,y,w,h = cv2.boundingRect(contour)
				if visualization:
					cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			else:
				count = 1
				for _, contour in enumerate(contours):
					area = cv2.contourArea(contour)
					if area>600 and area < 10000:
						self.boxls.append(contour)
						M = cv2.moments(contour)
						cx = int(M['m10']/M['m00'])
						cy = int(M['m01']/M['m00'])
						rect = cv2.minAreaRect(contour)
						box = np.int0(cv2.boxPoints(rect))
						x,y,w,h = cv2.boundingRect(contour)
						if visualization:
							cv2.circle(img, (cx, cy), 10, (0, 0, 255), 3)
							cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
							cv2.putText(img,str(count),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
						count += 1
		return box
		
	def warp(self, img):
		pts1 = np.float32([[153,120],[536,110],[65,457],[634,457]])
		pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		dst = cv2.warpPerspective(img,M,(640,480))
		return dst

		
	@staticmethod
	def SIFT(I, visualization=True):
		gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		descriptor = cv2.xfeatures2d.SIFT_create()
		kps, features = descriptor.detectAndCompute(gray, None)
		if visualization:
			cv2.drawKeypoints(I,kps,I,(0,255,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def main():
	cap=cv2.VideoCapture(0)
	sift = cv2.xfeatures2d.SIFT_create()
	while(1):
		object_detector(cap)
		if cv2.waitKey(5) & 0xFF == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()