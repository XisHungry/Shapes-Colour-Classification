# Detects all shapes
# Detects all colours
# Label shapes with its respective colours
# Only one kind of shape and colour at a time
# Use for reference

import cv2
import numpy as np
import math
import json

def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    __, frame = cap.read()

    frame = cv2.GaussianBlur(frame, (7, 7), 1.41)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edge = cv2.Canny(frame, 25, 75)

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
    lower_green = np.array([40, 52, 72])
    higher_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, higher_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Yellow color
    lower_yellow = np.array([20, 100, 20])
    higher_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, higher_yellow)
    yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    # Contours detection
    # Opencv version: 3.x.x & 4.x.x
    contours, _ = cv2.findContours(red_mask | blue_mask | yellow_mask | green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours1, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours3, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours4, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(0,len(contours)):
	       approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.01,True)
	       x = approx.ravel()[0]
	       y = approx.ravel()[1]
	       if(abs(cv2.contourArea(contours[i]))<3000 or not(cv2.isContourConvex(approx))):
		       continue

	       cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
	       if len (approx) >= 9: 
			if (contours1):
				cv2.putText(frame, "Red Circle", (x, y), font, 1, (255, 20, 147))
				print ("Red Circle")
	       		elif (contours2):
				cv2.putText(frame, "Blue Circle", (x, y), font, 1, (255, 20, 147))
				print ("Blue Circle")
	      		elif (contours3):
				cv2.putText(frame, "Green Circle", (x, y), font, 1, (255, 20, 147))
				print ("Green Circle")
	      		elif (contours4):
				cv2.putText(frame, "Yellow Circle", (x, y), font, 1, (255, 20, 147))
				print ("Yellow Circle")

    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edge)
    cv2.imshow("colour", red_mask | blue_mask | green_mask | yellow_mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
