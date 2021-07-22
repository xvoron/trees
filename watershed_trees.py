import cv2
import numpy as np
import matplotlib.pyplot as plt



frame = cv2.imread("doc/trees_rgb.jpg", 1)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_green = np.array([30, 30, 30])
upper_green = np.array([70, 255, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)
ret = cv2.bitwise_and(frame, frame, mask=mask)
gray = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.imshow("gray", gray)

#ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)

cv2.namedWindow("opening", cv2.WINDOW_NORMAL)
cv2.imshow("opening", opening)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

cv2.namedWindow("sure_bg", cv2.WINDOW_NORMAL)
cv2.imshow("sure_bg", sure_bg)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

cv2.namedWindow("dist_transform", cv2.WINDOW_NORMAL)
cv2.imshow("dist_transform", dist_transform)

ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

cv2.namedWindow("sure_fg", cv2.WINDOW_NORMAL)
cv2.imshow("sure_fg", sure_fg)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

cv2.waitKey(0)
cv2.destroyAllWindows()
