# Detects all shapes
# Detects all colours
# Label shapes with its respective colours
# Only one kind of shape and colour at a time
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
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Red color
	lower_red = np.array([130, 155, 84])
	higher_red = np.array([200, 255, 255])
	red_mask = cv2.inRange(hsv, lower_red, higher_red)
	red = cv2.bitwise_and(frame, frame, mask=red_mask)

	# Blue color
	lower_blue = np.array([94, 80, 2])
	higher_blue = np.array([126, 255, 255])
	blue_mask = cv2.inRange(hsv, lower_blue, higher_blue)
	blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

	# Green color
	lower_green = np.array([60, 52, 72])
	higher_green = np.array([100, 255, 255])
	green_mask = cv2.inRange(hsv, lower_green, higher_green)
	green = cv2.bitwise_and(frame, frame, mask=green_mask)

	# Yellow color
	lower_yellow = np.array([20, 100, 20])
	higher_yellow = np.array([40, 255, 255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, higher_yellow)
	yellow = cv2.bitwise_and(frame, frame, mask=green_mask)

	# Contours detection
	# Opencv version: 3.x.x & 4.x.x
	contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
	for i in range(0,len(contours)):
		approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.0075,True)
		x = approx.ravel()[0]
		y = approx.ravel()[1]
		if(abs(cv2.contourArea(contours[i]))<3000 or not(cv2.isContourConvex(approx))):
			continue

		cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
		if len (approx) == 3:
			if x > 300:
				cv2.putText(frame, "Right Triangle", (x, y), font, 1, (255, 0, 0))
				text_file =open("logs/Output.txt", "w")
				text_file.write("1")
				text_file.close()
			else:
				cv2.putText(frame, "Left Triangle", (x, y), font, 1, (255, 0, 0))
				text_file = open("logs/Output.txt", "w")
				text_file.write("2")
				text_file.close()
			break

		elif len (approx) == 4:
			if x > 300:
				cv2.putText(frame, "Right Rectangle", (x, y), font, 1, (255, 0, 0))
				text_file = open("logs/Output.txt", "w")				
				text_file.write("3")
				text_file.close()
			else:
				cv2.putText(frame, "Left Rectangle", (x, y), font, 1, (255, 0, 0))
				text_file = open("logs/Output.txt", "w")
				text_file.write("4")
				text_file.close()
				break

		elif len(approx) >= 9:
			if x > 300:
				cv2.putText(frame, "Right Circle", (x, y), font, 1, (255, 0, 0))
				text_file = open("logs/Output.txt", "w")
				text_file.write("5")
				text_file.close()
			else:
				cv2.putText(frame, "Left Circle", (x, y), font, 1, (255, 0, 0))
				text_file = open("logs/Output.txt", "w")
				text_file.write("6")
				text_file.close()
				break
			break

	cv2.imshow("Frame", frame)
	cv2.imshow("FrameG", green_mask)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()
