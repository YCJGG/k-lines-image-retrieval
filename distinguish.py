import numpy as np
import scipy.io as sio
import cv2
from scipy.spatial.distance import pdist
import time
from classfication import classify
from Preprocess import pre
import os


def judge(l):
    if l<= 1:
        return 2
    if l == 2:
        return 1
    if l > 2:
        return 0


# | refers to 2 
#  solid refers to 0
# empty refers to 1
def dis_content(flag,img):
    if flag == 1:
        l_1 = np.sum(img[5:415,105])//255
        c_1 = judge(l_1)
        return [c_1]


    elif flag == 2:
        
        l_1 = np.sum(img[5:415,55])//255
        l_2 = np.sum(img[5:415,350])//255
        c_1 = judge(l_1)
        c_2 = judge(l_2)
        return [c_1, c_2]
    elif flag == 3:
        l_1 = np.sum(img[5:415,35])//255
        l_2 = np.sum(img[5:415,235])//255
        l_3 = np.sum(img[5:415,435])//255
        c_1 = judge(l_1)
        c_2 = judge(l_2)   
        c_3 = judge(l_3)
        return [c_1,c_2,c_3]
    elif flag == 4:
        l_1 = np.sum(img[5:415,26])//255
        l_2 = np.sum(img[5:415,175])//255
        l_3 = np.sum(img[5:415,328])//255
        l_4 = np.sum(img[5:415,480])//255
        c_1 = judge(l_1)
        c_2 = judge(l_2)   
        c_3 = judge(l_3)
        c_4 = judge(l_4)
        return [c_1,c_2,c_3,c_4]
    elif flag == 5:
        l_1 = np.sum(img[5:415,18])//255
        l_2 = np.sum(img[5:415,142])//255
        l_3 = np.sum(img[5:415,264])//255
        l_4 = np.sum(img[5:415,386])//255
        l_5 = np.sum(img[5:415,508])//255
        c_1 = judge(l_1)
        c_2 = judge(l_2)   
        c_3 = judge(l_3)
        c_4 = judge(l_4)
        c_5 = judge(l_5)
        return [c_1,c_2,c_3,c_4,c_5]




# for _, _, content in os.walk('./5/'):
#     contents = content
# for cont in contents:
#     filename = './png/' + cont
#     # read a image and tranfer it to grayscale
#     img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#     img = pre(img)
#     a = dis_content(5,img)
#     print(a)