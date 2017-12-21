import os
import cv2
import numpy as np
import scipy.io as sio

def pre(img):
    ret, img = cv2.threshold(img, 12, 255,cv2.THRESH_BINARY)
    crop_img = img[60:480,100:720]
    crop_img[crop_img != 255] = 0
    crop_img = np.abs( 255 - crop_img )
    return crop_img