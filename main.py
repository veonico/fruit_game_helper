import utils
from PIL import ImageGrab
import cv2
import keyboard
import mouse
import numpy as np

bbox_locs = utils.get_canvas_position()

while True:
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox = bbox_locs)), cv2.COLOR_BGR2RGB)
    # print(type(image)) // np.ndarray
    cv2.imshow('image', image)
    key = cv2.waitKey(500)

    if key == ord("q"):
        break

cv2.destroyAllWindows()