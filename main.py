import utils
from PIL import ImageGrab
import cv2
import numpy as np
import recognizer
import mouse


bbox_locs = utils.get_canvas_position()
image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = bbox_locs)), cv2.COLOR_BGR2RGB)

# image = np.array(cv2.imread(r'./apple_images/apple_1.png'))
apple_vectors, boxes = utils.get_bboxes(image)
digit_matrix = recognizer.recognizer(apple_vectors)

blocked = [] # 이미 뽑혀서 처리가 된 것들

# digit matrix에서 10이 되는 경우의 수 하나 반환
num_iter = 0
padding = 10

while True:
    bbox = np.array(bbox_locs)
    num_iter += 1
    picked = utils.find_tens(digit_matrix)
    print(f"picked {num_iter} : {picked}")

    if picked:
        combined_boxes = utils.combine_picked(picked, boxes)

        # 이미 뽑힌 것들 block 처리
        if blocked:
            start_point, end_point = blocked.pop()
            cv2.rectangle(image, start_point, end_point, (255, 255, 255), -1)

        # 10의 되는 경우의 수의 bounding-box 표시하기
        for combined_box in combined_boxes:
            start_point = np.array(combined_box[:2])
            end_point = np.array(combined_box[2:])

            blocked.append((start_point, end_point))

            cv2.rectangle(image, start_point, end_point, (0, 255, 255), 2)

            start_point = bbox_locs[:2] + start_point - padding
            end_point = bbox_locs[:2] + end_point + padding
            
            mouse.move(*start_point)
            mouse.drag(*start_point, *end_point, duration = 0.5)

        cv2.imshow('image', image)
        cease_key = cv2.waitKey(500)

        if cease_key == ord('q'):
            print("quit")
            break
    else:
        print("No Combination detected")
        break

    if num_iter == 3:
        break

cv2.destroyAllWindows()