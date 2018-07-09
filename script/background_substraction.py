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
		self.cloud_des = []

		_, origin = cap.read()
		rect = self.camrectify(origin)
		cv2.imshow('res', rect)
		#-------------warp the image---------------------#
		warp = self.warp(origin)
		#-------------segment the object----------------#
		hsv = cv2.cvtColor(warp,cv2.COLOR_BGR2HSV)        
		green_mask = cv2.inRange(hsv, np.array([57,145,0]), np.array([85,255,255]))
		hand_mask = cv2.inRange(hsv, np.array([84,32,0]), np.array([153,255,255]))
		hand_mask = cv2.dilate(hand_mask, kernel = np.ones((5,5),np.uint8))
		res=cv2.bitwise_and(warp, warp, mask = green_mask)
		thresh = 255 - green_mask
		thresh = cv2.subtract(thresh, hand_mask)
		# cv2.imshow('res', thresh)
		# cv2.imshow('res1', hand_mask)
		# cv2.imshow('res2', object_mask)
		# object_mask = cv2.subtract(warp, res)
		# gray = cv2.cvtColor(object_mask,cv2.COLOR_BGR2GRAY)
		# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		draw_img = warp.copy()
		#-------------get the bounding box--------------
		self.get_minRect(draw_img, thresh, only=False, visualization=False)
		#-------------save descriptor----------------#
		if save:
			self.get_descriptor(draw_img, meth='SURF')
		#-------------retrieve descriptor------------------
		else: 
			self.check_descriptor(draw_img, meth='SURF')
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
		
	def warp(self, img):
		pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
		pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		dst = cv2.warpPerspective(img,M,(640,480))
		return dst

	@staticmethod
	def detect(frame, method='SURF', visualization=True):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if method == 'SURF':
			descriptor = cv2.xfeatures2d.SURF_create()
		if method == 'SIFT':
			descriptor = cv2.xfeatures2d.SIFT_create()
		if method == 'ORB':
			descriptor = cv2.ORB_create(nfeatures=100000,scoreType=cv2.ORB_FAST_SCORE)
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
	
	@staticmethod
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
	
	def get_descriptor(self, frame, meth): 
		temp_array = []
		for i in range(len(self.boxls)): 
			x,y,w,h = cv2.boundingRect(self.boxls[i])
			temp = frame[y:y+h, x:x+w, :]
			kps, features = self.detect(temp, method=meth)
			temp_array.append(self.pickle_keypoints(kps, features))
			#cv2.imshow(str(i), temp)
		pickle.dump(temp_array, open("keypoints_database.p", "wb"))
		
			#print("features:{} in {}".format(features, i))
	
	def check_descriptor(self, frame, meth): 
		keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ))
		for i in range(len(self.boxls)): 
			x,y,w,h = cv2.boundingRect(self.boxls[i])
			temp = frame[y:y+h, x:x+w, :]
			_, des1 = self.detect(temp, method=meth)
			
			ind = 0
			temp = 0
			
			for j in range(len(keypoints_database)): 
				_, des2 = self.unpickle_keypoints(keypoints_database[j])

				if meth == 'ORB': 
					# create BFMatcher object
					bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
					# Match descriptors.
					matches = bf.match(des1,des2)
					temp = [q.distance for q in matches]
					good_len = sum([p.distance < 10 for p in matches])
					if good_len > temp: 
						ind = j
						temp = good_len
				else:
					FLANN_INDEX_KDTREE = 0
					index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
					search_params = dict(checks=50)   # or pass empty dictionary

					flann = cv2.FlannBasedMatcher(index_params,search_params)
					# print('des1:{}, des2:{}'.format(des1, des2))
					matches = flann.knnMatch(des1,des2,k=2)
					goodMatch=[]
					# ratio test as per Lowe's paper
					for k,(m,n) in enumerate(matches):
						if m.distance < 0.7*n.distance:
							goodMatch.append(m)
					# print(i, j, len(goodMatch))
					if len(goodMatch) > temp: 
						ind = j
						temp = len(goodMatch)
			# ---------------------------visualization---------------------------
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
			cv2.putText(frame, str(ind),(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,0))
			




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
