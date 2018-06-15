import cv2
from object_tracking import object_tracking
from hand_tracking import hand_tracking


def main():
	cap=cv2.VideoCapture(0)
	while(1):
		object_tracking(cap)
		hand_tracking(cap)
		k = cv2.waitKey(5) & 0xFF
		if k == 27:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()