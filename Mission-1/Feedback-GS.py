# Detects Green Square
# Send feedback to a Text file
# Use for SAFMC

import cv2
import numpy as np
import math

def nothing(x):
	# any operation
	pass

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

while True:
	__, frame = cap.read()

	#convert to gray and edges
	frame = cv2.GaussianBlur(frame, (7, 7), 1.41)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Green color
	lower_green = np.array([60, 52, 72])
	upper_green = np.array([100, 255, 255])
	green_mask = cv2.inRange(hsv, lower_green, upper_green)
	green = cv2.bitwise_and(frame, frame, mask=green_mask)

	# Contours detection
	# Opencv version: 3.x.x & 4.x.x
	contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
	for i in range(0,len(contours)):
		approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.0075,True)
		x = approx.ravel()[0]
		y = approx.ravel()[1]
		if(abs(cv2.contourArea(contours[i]))<1000 or not(cv2.isContourConvex(approx))):
			continue

		if len (approx) == 4:
			cv2.putText(green_mask, "Green Square", (x, y), font, 1, (255, 0, 0))
			text_file = open("logs/Output.txt", "w")
			text_file.write("1")
			text_file.close()



	cv2.imshow("Frame", frame)
	cv2.imshow("Green Only", green_mask)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()
