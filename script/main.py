import cv2
from object_tracking import object_tracking
from hand_tracking import hand_tracking
import numpy as np
import logging

SCALAR = 1

def paint(cap, cir, cnt, hand_cnt): 
	logger = logging.getLogger('ftpuploader')
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * SCALAR
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * SCALAR
	print width, height

	suf = np.zeros((height, width, 3), np.uint8)
	suf[suf == 0] = 255

	for k in cir: 
		color, circle = cir[k]
		for c in circle: 
			cv2.circle(suf,(c[0] * SCALAR ,c[1] * SCALAR), 10, color,-1)
	for k in cnt: 
		try:
			color = cnt[k][0]
			contour = cnt[k][1][0] * SCALAR
			cv2.drawContours(suf,[contour],0,color,-1)
		except Exception, e:
			logger.error('Failed to upload to ftp: '+ str(e))
	for k in hand_cnt: 
		#cnt = hand_cnt[k]
		cv2.drawContours(suf,k,-1,(0, 255, 0), -1)
	# suf = cv2.resize(suf, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	
	####################set the full screen#######################
	# cv2.namedWindow("surface", cv2.WND_PROP_FULLSCREEN)
	# cv2.setWindowProperty("surface",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

	cv2.imshow('surface', suf)
	#print suf

def main(): 
	
	cap=cv2.VideoCapture(0)	
	while(1):
		obj = object_tracking(cap)
		cir, cnt = obj.get_result()
		hand = hand_tracking(cap)
		hand_cnt = hand.get_result()
		paint(cap, cir, cnt, hand_cnt)
		if cv2.waitKey(5) & 0xFF == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()