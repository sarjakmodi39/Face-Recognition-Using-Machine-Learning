import cv2
import numpy as np

def nothing(x):
	pass

cap=cv2.VideoCapture(0)
faceCascade=cv2.CascadeClassifier('cascade/Data/haarcascade_frontalface_alt2.xml')
cv2.namedWindow("frame")
cv2.createTrackbar("Neighbours","frame",5,20,nothing)

while True:
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	neighbours=cv2.getTrackbarPos("Neighbours","frame")

	faces=faceCascade.detectMultiScale(gray,1.1,neighbours)
		# minSize=(20,20)
	
	# print(faces)	
	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x, y),(x+w,y+h),(0,255,0),2)

	cv2.imshow('frame',frame)
	
	k=cv2.waitKey(30) & 0xff
	if k==27:
		break

cap.release()
cv2.destroyAllWindows()