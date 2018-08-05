from __future__ import print_function
import numpy as np
import cv2
import time
import pickle
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.externals import joblib
import time

toarr = lambda x, y, z : np.array([x, y, z], np.uint8)
mean_ = lambda x : np.sum(x) // np.count_nonzero(x)


class object_detector(): 
	def __init__(self, start): 		
		self.start_time = start
		self.stored_flag = False
		self.trained_flag = False
		self.boxls = None
		self.cloud_des = []
		self.cloud_des_new = None
		self.collect_count = None

	def update(self, cap, save=False, train=False):
		self.boxls = []
		# self.cloud_des = []
		self.cloud_des_new = []
		self.collect_count = 0
		_, origin = cap.read()
		rect = self.camrectify(origin)

		#-------------warp the image---------------------#
		warp = self.warp(rect)
		#-------------segment the object----------------#
		hsv = cv2.cvtColor(warp,cv2.COLOR_BGR2HSV)
		green_mask = cv2.inRange(hsv, np.array([57,145,0]), np.array([85,255,255]))
		hand_mask = cv2.inRange(hsv, np.array([118,32,0]), np.array([153,255,255]))
		hand_mask = cv2.dilate(hand_mask, kernel = np.ones((7,7),np.uint8))

		skin_mask = cv2.inRange(hsv, np.array([0,52,0]), np.array([56,255,255]))
		skin_mask = cv2.dilate(skin_mask, kernel = np.ones((5,5),np.uint8))

		thresh = 255 - green_mask
		thresh = cv2.subtract(thresh, hand_mask)
		thresh = cv2.subtract(thresh, skin_mask)

		draw_img = warp.copy()
		self.train_img = warp.copy()
		#-------------get the bounding box--------------
		self.get_minRect(draw_img, thresh, only=False, visualization=True)
		#--------------get bags of words and training-------#
		if not self.stored_flag:
			self.stored_flag = self.store(10.0)
		if self.stored_flag and not self.trained_flag: 
			self.trained_flag = self.train()
		if self.trained_flag: 
			self.predict()


		#-------------save descriptor----------------#
		# if save:
		# 	self.get_descriptor(draw_img, meth='SURF')
			# self.train(train_img)
		#-------------retrieve descriptor------------------
		# else:
		# 	self.check_descriptor(draw_img, meth='SURF')
		# cv2.imshow('green_mask', draw_img)


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
					cv2.rectangle(img,(x,y),(x+w,y+h))
			else:
				for _, contour in enumerate(contours):
					area = cv2.contourArea(contour)
					if area>600 and area < 10000:					
						M = cv2.moments(contour)
						cx = int(M['m10']/M['m00'])
						cy = int(M['m01']/M['m00'])
						rect = cv2.minAreaRect(contour)
						box = np.int0(cv2.boxPoints(rect))
						x,y,w,h = cv2.boundingRect(contour)
						self.boxls.append([x, y, w, h])
				#---------------sorting the list according to the x coordinate of each item
				if len(self.boxls) > 0:
					boxls_arr = np.array(self.boxls)
					self.boxls = boxls_arr[boxls_arr[:, 0].argsort()].tolist()
				for i in range(len(self.boxls)): 
					if visualization: 
						x,y,w,h = self.boxls[i]
						cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
						cv2.putText(img,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,0,255))
		cv2.imshow('img', img)


	def warp(self, img):
		pts1 = np.float32([[115,124],[520,112],[2,476],[640,480]])
		pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		dst = cv2.warpPerspective(img,M,(640,480))
		return dst

	def get_descriptor(self, frame, meth):
		temp_array = []
		for i in range(len(self.boxls)):
			x,y,w,h = self.boxls[i]
			temp = frame[y:y+h, x:x+w, :]
			kps, features = self.detect(temp, method=meth)
			temp_array.append(self.pickle_keypoints(kps, features))
		pickle.dump(temp_array, open("keypoints_database.p", "wb"))


	def check_descriptor(self, frame, meth, visualization=True):
		keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ))
		occupy_ls = []
		for i in range(len(self.boxls)):
			x,y,w,h = self.boxls[i]
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
					if len(goodMatch) > temp and j not in occupy_ls:
						ind = j
						temp = len(goodMatch)
			if ind not in occupy_ls:
				occupy_ls.append(ind)
				print(occupy_ls)
				# ---------------------------visualization---------------------------
				if visualization:
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
					cv2.putText(frame, str(ind),(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,0))

	def store(self, store_time):
		frame = self.train_img
		num_object = len(self.boxls)
		#-------------capturing img for each of item--------------#
		for i in range(num_object):
			x,y,w,h = self.boxls[i]
			temp = frame[y:y+h, x:x+w, :]
			_, features = self.detect(temp)
			self.cloud_des.append(features)
		self.collect_count += 1
		if time.time() - self.start_time < store_time: 
			print('still collecting--------')
			return False
		else: 
			if len(self.cloud_des) == num_object: 
				for i in range(num_object): 
					self.cloud_des_new.append([self.cloud_des[i]])

			else: 
				for i in range(num_object):
					temp_mask = [k for k in range(i, len(self.cloud_des), num_object)]
					print('temp_mask', temp_mask)
					temp = np.asarray(self.cloud_des)[np.asarray(temp_mask)].tolist()
					self.cloud_des_new.append(temp)
			print('finish collecting')
			return True

	def train(self):
		cloud_des = self.cloud_des_new
		#print(len(self.cloud_des_new))
		label = []
		bow = []
		#--------------create labels----------------------------#
		for i in range(len(cloud_des)):
			for j in range(len(cloud_des[i])):
				label.append(i)
		for i in range(len(cloud_des)):
			num_img = len(cloud_des[i])
			#------------create des as descriptor matrix--------#
			des = cloud_des[i][0]
			for j in range(1, num_img):
				des = np.vstack((des, cloud_des[i][j]))
			#-----------create bag of words-----------------#
			k = 5
			voc, _ = kmeans(des, k, 1)
			im_features = np.zeros((num_img, k), "float64")
			for j in range(num_img):
				words, _ = vq(cloud_des[i][j],voc)
				for w in words:
					im_features[j][w] += 1
			stdSlr = StandardScaler().fit(im_features)
			im_features = stdSlr.transform(im_features)

			for i in range(im_features.shape[0]):
				bow.append(im_features[i])
		#------------------------training------------------------#
		print('start training')
		clf = svm.SVC(decision_function_shape='ovo')
		clf.fit(bow, label)
		print("complete fit")
		joblib.dump((clf, stdSlr, k, voc), "bof.pkl", compress=3)    
		return True

	def predict(self): 
		num_object = len(self.boxls) 
		clf, stdSlr, k, voc = joblib.load("bof.pkl")
		frame = self.train_img
		test_features = np.zeros((num_object, k), "float32")
		for i in range(num_object):
			x,y,w,h = self.boxls[i]
			temp = frame[y:y+h, x:x+w, :]
			_, des1 = self.detect(temp)
			stdSlr = StandardScaler().fit(test_features)
			test_features = stdSlr.transform(test_features)
			result = clf.predict(test_features)
			print(result)
			try:
				words, _ = vq(des1,voc)
				for w in words:
					test_features[i][w] += 1
			except Exception: 
				print(Exception)
			
		
		








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






def main():
	cap=cv2.VideoCapture(0)
	# object_detector(cap, save=True)
	start_time = time.time()
	detector = object_detector(start_time)
	while(1):
		detector.update(cap, save=True)
		if cv2.waitKey(5) & 0xFF == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
