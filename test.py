from PIL import Image
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import KMeans

img = cv2.cvtColor(np.array(Image.open('apple.png')), cv2.COLOR_RGB2BGR)

# cv2.imshow('apple', img)

red_lower = np.array([17, 15, 100], dtype = "uint8")
red_upper = np.array([50, 56, 255], dtype = "uint8")

mask = cv2.inRange(img, red_lower, red_upper) # 흑백 
# output = cv2.bitwise_and(img, img, mask=mask) # 색 입힌 거

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []

# 상자 검출
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    boxes.append([x, y, x+w, y+h])  # top-left, bottom-right

imgs = []

for box_idx, box in enumerate(boxes):
    tx1, ty1, tx2, ty2  = box
    temp_img = cv2.resize(img[ty1:ty2, tx1:tx2, :], (10, 10), interpolation=cv2.INTER_AREA)

    imgs.append(temp_img.flatten())

imgs = np.concatenate([imgs])

center_path = "cluster_centers.csv"
center_data = pd.read_csv(center_path, index_col = None).values

kmeans = KMeans(n_clusters = 9, init = center_data, n_init = 1, random_state = 0).fit(imgs)
answer = kmeans.predict(imgs)

print(answer)

cv2.imshow("temp", imgs[1].reshape([10, 10, -1]))
cv2.waitKey(0)



