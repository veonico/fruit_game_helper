import utils
from PIL import ImageGrab
import cv2
import numpy as np
import recognizer

bbox_locs = utils.get_canvas_position()

# 최초 digit recognition

image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = bbox_locs)), cv2.COLOR_BGR2RGB)
# image = np.array(cv2.imread(r'./apple_images/apple.png'))
apple_vectors, boxes = utils.get_bboxes(image)
# print(boxes)
digit_matrix = recognizer.recognizer(apple_vectors)

picked = utils.find_nine_ones(digit_matrix)
combined_boxes = utils.combine_picked(picked, boxes)

print(combined_boxes)

for combined_box in combined_boxes:
    start_point = combined_box[:2]
    end_point = combined_box[2:]
    cv2.rectangle(image, start_point, end_point, (0, 255, 255), 2)

cv2.imshow('image', image)
cv2.waitKey(0)

# while True:
#     image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = bbox_locs)), cv2.COLOR_BGR2RGB)
#     # print(type(image)) // np.ndarray
#     cv2.imshow('image', image)
#     key = cv2.waitKey(500)

#     if key == ord("q"):
#         break

# cv2.destroyAllWindows()