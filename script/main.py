import cv2
from object_tracking import object_tracking
from hand_tracking import hand_tracking
import numpy as np
import logging

def paint(cap, cir, cnt, hand_cnt): 
	logger = logging.getLogger('ftpuploader')
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	suf = np.zeros((height, width, 3), np.uint8)

	for k in cir: 
		color, circle = cir[k]
		for c in circle: 
			cv2.circle(suf,(c[0],c[1]),c[2],color,-1)
		for k in cnt: 
			try:
				color = cnt[k][0]
				contour = cnt[k][1]
				cv2.drawContours(suf,contour,0,color,-1)
			except Exception, e:
				logger.error('Failed to upload to ftp: '+ str(e))
		
		cv2.imshow('surface', suf)

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