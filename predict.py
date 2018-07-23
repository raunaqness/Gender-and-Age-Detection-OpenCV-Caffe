import cv2
import imutils
import time
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(3, 480) #set width
cap.set(4, 640) #set height

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def get_models():
	pass

def read_from_camera():
	while True:

		ret, image = cap.read()

		face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)

		print("Found {} faces".format(str(len(faces))))

		for (x, y, w, h )in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)


		cv2.imshow('frame', image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	read_from_camera()