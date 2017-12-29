import os
import cv2
import numpy as np
import scipy.io as sio
from classfication import classify

images = []
crop_images =[]

#classfication
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []

#crop images classfication
c_l1 = []
c_l2 = []
c_l3 = []
c_l4 = []
c_l5 = []

c_l=[c_l1,c_l2,c_l3,c_l4,c_l5]
l = [l1, l2, l3, l4, l5]

K = 15*255
path = './png/'

for _, _, content in os.walk(path):
    contents = content
   
for cont in contents:
    filename = path + cont
    # read a image and tranfer it to grayscale
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # transfer to binary image
    ret, img = cv2.threshold(img, 77, 255,cv2.THRESH_BINARY)
    
   
    #original 600 800
    #crop 420 620
    
    crop_img = img[60:480,100:720]
    
    crop_img[crop_img != 255] = 0
    img[img != 255] = 0
    
    #reverse
    crop_img = np.abs( 255 - crop_img )


    category = classify(K,crop_img)
   
    img_path = './'+str(category)
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    cv2.imwrite(img_path+'/'+cont,img)
    l[category-1].append(img)
    c_l[category-1].append(crop_img)
 
    images.append(img)
    crop_images.append(crop_img)

images = np.array(images)
crop_images = np.array(crop_images)
l = [ np.array(x) for x in l]
c_l = [np.array(x) for x in c_l]

for index in range(len(l)):
    sio.savemat('image_data_'+str(index),{'images':l[index],'crop_images':c_l[index]})


sio.savemat('image_data',{'images':images,'crop_images':crop_images})

