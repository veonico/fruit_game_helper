from PIL import ImageGrab
import cv2
import mouse
import numpy as np


def get_canvas_position():
    """
    게임 화면의 top-right corner과 bottom-left corner의 좌표를 manual하게 찍어, 해당 canvas의 위치를 반환한다.
    """

    x_left = 0
    y_top = 0
    x_right = 0
    y_bottom = 0

    print("click the left-top corner of the game canvas")

    while not x_left * y_top:
        while not mouse.is_pressed():
            x_left, y_top = mouse.get_position()

    print("click the right-bottom corner of the game canvas")

    while not x_right * y_bottom:
        while not mouse.is_pressed():
            x_right, y_bottom = mouse.get_position()

    print(f"ROI : ({x_left}, {y_top}, {x_right}, {y_bottom})")

    return x_left, y_top, x_right, y_bottom
    

def get_bboxes(img):
    """
    이미지를 받아, 해당 이미지에서 숫자를 포함하는 bounding box를 이미지로 변환 후, 이미지들의 집합을 반환한다.
    """

    red_lower = np.array([17, 15, 100], dtype = "uint8")
    red_upper = np.array([50, 56, 255], dtype = "uint8")

    mask = cv2.inRange(img, red_lower, red_upper) # 흑백 

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    # 상자 검출
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x+w, y+h])  # top-left, bottom-right


    imgs = []

    for box in boxes:
        tx1, ty1, tx2, ty2  = box
        temp_img = cv2.resize(img[ty1:ty2, tx1:tx2, :], (10, 10), interpolation=cv2.INTER_AREA)

        imgs.append(temp_img.flatten())

    imgs = np.concatenate([imgs])

    # 상자 뒤집기 (뒤에서부터 탐색하기 때문)
    boxes = np.flip(np.array(boxes), axis = 0).reshape([10, 17, -1])

    return imgs, boxes

def find_nine_ones(digit_matrix):
    """
    숫자로 이루어진 배열을 받아, 해당 배열 내에서 (9-1) 조합을 찾는다.
    """

    ROW_MAX = 9
    COL_MAX = 16

    nine_locs = np.concatenate([np.where(digit_matrix == 9)]).T
    picked = []

    delta = [(0,1), (0, -1), (1,0), (-1, 0)]
    
    for row_idx, col_idx in nine_locs:
        for row_off, col_off in delta:
            new_row = row_idx + row_off
            new_col = col_idx + col_off

            if 0 <= new_row < ROW_MAX and 0 <= new_col < COL_MAX:
                if digit_matrix[new_row][new_col] == 1:
                    picked.append((row_idx, col_idx, new_row, new_col))
                    break

    return picked

def combine_picked(picked, boxes):
    """
    선정된 bounding box를 합쳐 하나의 bounding box로 만든다.
    """
    combined_boxes = []

    for picked_one in picked:
        r1, c1, r2, c2 = picked_one

        x11, y11, x12, y12 = boxes[r1][c1]
        x21, y21, x22, y22 = boxes[r2][c2]

        x1 = min(x11, x21)
        y1 = min(y11, y21)
        x2 = max(x12, x22)
        y2 = max(y12, y22)

        combined_boxes.append((x1, y1, x2, y2))

    return combined_boxes

