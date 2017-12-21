import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
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

for _, _, content in os.walk('./png/'):
    contents = content
   
for cont in contents:
    filename = './png/' + cont
    # read a image and tranfer it to grayscale
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # transfer to binary image
    ret, img = cv2.threshold(img, 12, 255,cv2.THRESH_BINARY)
    
    #cv2.imshow('r',img)
    #cv2.waitKey(0)
    #original 600 800
    #crop 420 620
    #print(img.shape)
    crop_img = img[60:480,100:720]
    
    #binaryzation , 1 refers to 255
    crop_img[crop_img != 255] = 0
    img[img != 255] = 0
    #crop_img[crop_img == 255] = 0
    # img[img == 255] = 1

    # test to show images
    crop_img = np.abs( 255 - crop_img )
    #cv2.imshow('ori',img)
    #cv2.imshow(filename,crop_img)
    #print(crop_img)
    #cv2.waitKey(0)
    
    #classfication
    # method and tables
    # # 1 line  55
    # # 2 lines 30  325
    # # 3 lines 20  220  420
    # # 4 lines 15  167  318  470
    # # 5 lines 12  134  256  378   500
    
    # judge the midline of k lines +-2

    # first prepare the features
    # 5 lines
    # l5_1 =  np.sum(crop_img[:,10:15])
    # l5_2 =  np.sum(crop_img[:,132:137])
    # l5_3 =  np.sum(crop_img[:,254:259])
    # l5_4 =  np.sum(crop_img[:,376:381])
    # l5_5 =  np.sum(crop_img[:,498:503])
    # # 4 lines
    # l4_1 =  np.sum(crop_img[:,13:18])
    # l4_2 =  np.sum(crop_img[:,165:170])
    # l4_3 =  np.sum(crop_img[:,316:321])
    # l4_4 =  np.sum(crop_img[:,468:473])
    # # 3 lines
    # l3_1 =  np.sum(crop_img[:,18:23])
    # l3_2 =  np.sum(crop_img[:,218:223])
    # l3_3 =  np.sum(crop_img[:,418:423])
    # #2 lines
    # l2_1 =  np.sum(crop_img[:,28:33])
    # l2_2 =  np.sum(crop_img[:,323:328])
    # # 1 lines 
    # l_1 = np.sum(crop_img[:,53:58])
    # #cv2.imshow('ori',img)
    # #cv2.waitKey(0)

    # #print(l_1)
    # #classfication
    # if (l5_1>K) and (l5_2>K) and (l5_3>K) and (l5_4>K) and (l5_5>K):
    #     c_l5.append(crop_img)
    #     l5.append(img)  
    #     cv2.imwrite('./5/'+cont,img)
    # elif (l4_1>K) and (l4_2>K) and (l4_3>K) and (l4_4>K):
    #     c_l4.append(crop_img)
    #     l4.append(img)  
    #     cv2.imwrite('./4/'+cont,img)
    # elif (l3_1>K) and (l3_2>K) and (l3_3>K) :
    #     c_l3.append(crop_img)
    #     l3.append(img)  
    #     cv2.imwrite('./3/'+cont,img)
    # elif (l2_1>K) and (l2_2>K):
    #     c_l2.append(crop_img)
    #     l2.append(img)  
    #     cv2.imwrite('./2/'+cont,img)
    # elif (l_1>K):
    #     c_l1.append(crop_img)
    #     l1.append(img)  
    #     cv2.imwrite('./1/'+cont,img)

    category = classify(K,crop_img)
    #print(category)
    cv2.imwrite('./'+str(category)+'/'+cont,img)
    l[category-1].append(img)
    c_l[category-1].append(crop_img)
 
    images.append(img)
    crop_images.append(crop_img)

images = np.array(images)
crop_images = np.array(crop_images)
l = [ np.array(x) for x in l]
c_l = [np.array(x) for x in c_l]
#l = np.array(l)
#c_l = np.array(c_l)
for index in range(len(l)):
    sio.savemat('image_data_'+str(index),{'images':l[index],'crop_images':c_l[index]})

#np.save('images.npy',images)
#np.save('crop_images.npy',crop_images)
sio.savemat('image_data',{'images':images,'crop_images':crop_images})

