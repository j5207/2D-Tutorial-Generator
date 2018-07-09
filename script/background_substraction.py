from __future__ import print_function
import numpy as np
import cv2
import time
import pickle

toarr = lambda x, y, z : np.array([x, y, z], np.uint8)
mean_ = lambda x : np.sum(x) // np.count_nonzero(x)

class object_detector():
	def __init__(self, cap, save=False):
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
		#-------------save descriptor----------------#
		if save:
			self.get_descriptor(draw_img)
		#-------------retrieve descriptor------------------
		else: 
			self.check_descriptor(draw_img)
			# keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ))
			# for i in range(len(keypoints_database)): 
			# 	kps, desc = self.unpickle_keypoints(keypoints_database[i])
				#------------------matching------------------
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
				# count = 1
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
						# 	cv2.putText(img,str(count),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
						# count += 1
		return box
		
	def warp(self, img):
		pts1 = np.float32([[153,120],[536,110],[65,457],[634,457]])
		pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		dst = cv2.warpPerspective(img,M,(640,480))
		return dst

		
	@staticmethod
	def SIFT(frame, visualization=True):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		descriptor = cv2.xfeatures2d.SIFT_create()
		kps, features = descriptor.detectAndCompute(gray, None)
		if visualization:
			cv2.drawKeypoints(frame,kps,frame,(0,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		return kps, features

	@staticmethod
	def pickle_keypoints(keypoints, descriptors):
		temp_array = []
		for i, point in enumerate(keypoints):
			temp = (point.pt, point.size, point.angle, point.response, point.octave,
			point.class_id, descriptors[i])
			temp_array.append(temp)
		return temp_array

	@staticmethod
	def unpickle_keypoints(array):
		keypoints = []
		descriptors = []
		for point in array: 
			temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
			temp_descriptor = point[6]
			keypoints.append(temp_feature)
			descriptors.append(temp_descriptor)
		return keypoints, np.array(descriptors)
	
	def get_descriptor(self, frame): 
		temp_array = []
		for i in range(len(self.boxls)): 
			x,y,w,h = cv2.boundingRect(self.boxls[i])
			temp = frame[y:y+h, x:x+w, :]
			kps, features = self.SIFT(temp)
			temp_array.append(self.pickle_keypoints(kps, features))
			#cv2.imshow(str(i), temp)
		pickle.dump(temp_array, open("keypoints_database.p", "wb"))
			#print("features:{} in {}".format(features, i))
	
	def check_descriptor(self, frame): 
		keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ))
		for i in range(len(self.boxls)): 
			x,y,w,h = cv2.boundingRect(self.boxls[i])
			temp = frame[y:y+h, x:x+w, :]
			_, des1 = self.SIFT(temp)
			
			ind = 0
			temp = 0
			
			for j in range(len(keypoints_database)): 
				_, des2 = self.unpickle_keypoints(keypoints_database[j])
				# FLANN parameters
				FLANN_INDEX_KDTREE = 0
				index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
				search_params = dict(checks=50)   # or pass empty dictionary

				flann = cv2.FlannBasedMatcher(index_params,search_params)

				matches = flann.knnMatch(des1,des2,k=2)
				print(i, j, len(matches))

				goodMatch=[]

				# ratio test as per Lowe's paper
				for k,(m,n) in enumerate(matches):
					if m.distance < 0.75*n.distance:
						goodMatch.append(m)

				if len(goodMatch) > temp: 
					ind = j
					temp = len(goodMatch)
				# print(ind, temp)
			cv2.putText(frame, str(ind),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
			




def main():
	cap=cv2.VideoCapture(0)
	object_detector(cap, save=True)
	while(1):
		object_detector(cap)
		if cv2.waitKey(5) & 0xFF == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
