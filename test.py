from PIL import Image
import numpy as np
import cv2

img = cv2.cvtColor(np.array(Image.open('apple.png')), cv2.COLOR_RGB2BGR)

# cv2.imshow('apple', img)

red_lower = np.array([17, 15, 100], dtype = "uint8")
red_upper = np.array([50, 56, 255], dtype = "uint8")

mask = cv2.inRange(img, red_lower, red_upper) # 흑백 
# output = cv2.bitwise_and(img, img, mask=mask) # 색 입힌 거

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    boxes.append([x, y, w, h])

for box in boxes:
    top_left     = (box[0], box[1])
    bottom_right = (box[0] + box[2], box[1] + box[3])
    cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)

cv2.imshow('apple', img)
cv2.waitKey(1000)

