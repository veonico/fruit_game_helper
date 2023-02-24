from PIL import ImageGrab
import cv2
import keyboard
import mouse
import numpy as np


# 캔버스의 위치 가져오기
def get_canvas_position():

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
    

def initialize():
    # 스크린 모니터링
    bbox_locs = get_canvas_position()

    while True:
        image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = bbox_locs)), cv2.COLOR_BGR2RGB)
        cv2.imshow('image', image)
        key = cv2.waitKey(100)

        if key == ord("q"):
            break
    
    cv2.destroyAllWindows()

