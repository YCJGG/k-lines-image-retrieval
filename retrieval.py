import numpy as np
import scipy.io as sio
import cv2
from scipy.spatial.distance import pdist
import time

image_data = sio.loadmat('image_data.mat')
ori_images = image_data['images']
crop_images = image_data['crop_images']

#ori_images[ori_images == 1] = 255

# for i in range(4):
#     cv2.imshow('ori',np.squeeze(ori_images[i,:,:]))
#     cv2.waitKey(0)

#crop_images = np.abs(crop_images-1)
#crop_images[crop_images == 255] = 1
query_images = crop_images[0:4,:,:]
crop_images = crop_images[4:,:,:]
retrival_images = ori_images[4:,:,:]

# num of query_images
N = query_images.shape[0]
# num of images
N_ = crop_images.shape[0]

#img = np.squeeze(crop_images[0,:,:])
#cv2.imshow('r',img)
#cv2.waitKey(10000)

print(crop_images.shape)

def Cos_dis(x,y):
   pass 

def Jaccard_dis(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'jaccard')
    return d2

def Hammimg_dis(x,y):
    pass 

# shape (N,w*h) 
#crop_images  = crop_images.reshape([N_, -1])

# # Hamming_dis
# crop = crop_images
# for i in range(N):
#     query = query_images[i,:,:]
#     #query = query.reshape([1,-1])
#     H = crop - query
#     H = np.abs(H)
#     H = np.mean(H,(1,2))
#     print(np.min(H))
#     index = np.argwhere(H == np.min(H))
#     for idx in index:
#         idx = idx[0]
#         img = retrival_images[idx,:,:]
#         img = np.squeeze(img)
#         cv2.imshow(str(i),img)
#         cv2.waitKey(0)

query = query_images[3,:,:]
query = np.squeeze(query)
feature_1 = query[20:380,0:50].reshape(-1)

feature_2 = query[20:380,195:245].reshape(-1)
feature_3 = query[20:380,395:445].reshape(-1)

# cv2.imshow('f',feature_1)
# cv2.waitKey(0)
#query = query.reshape(-1)
dis = []
for i in range(N_):
   
    r = crop_images[i,:,:]
    r = np.squeeze(r)
    r_1 = r[20:380,0:50].reshape(-1)
    r_2 = r[20:380,195:245].reshape(-1)
    r_3 = r[20:380,395:445].reshape(-1)
    s = time.time()
    d1 = Jaccard_dis(feature_1,r_1)
    d2 = Jaccard_dis(feature_2,r_2)
    d3 = Jaccard_dis(feature_3,r_3)
    d = (d1[0]+d2[0]+d3[0])/3

    dis.append(d)
    e = time.time()
    print(i,e-s)
print(dis)
dis = np.array(dis)
dis = np.argsort(dis)
print(dis[0:10])

for idx in dis[0:10]:
    img = retrival_images[idx,:,:]
    img = np.squeeze(img)
    cv2.imshow(str(idx),img)
    cv2.waitKey(0)


#todo 
#对图片切片，在三个维度上进行比较，然后相似性加权
