import numpy as np
import scipy.io as sio
import cv2
from scipy.spatial.distance import pdist
import time
from classfication import classify
from Preprocess import pre

K = 15*255

def load_data(flag):
    flag = flag - 1 
    data = sio.loadmat('image_data_'+str(flag)+'.mat')
    images = data['images']
    crop_images = data['crop_images']

    return images, crop_images

def Jaccard_dis(x,y):
    X=np.vstack([x,y])
    d2=pdist(X,'jaccard')
    return d2

def get_features(flag,img):
    if flag == 1:
        return [img[20:380,0:110].reshape(-1)]
    elif flag == 2:
        return[img[20:380,0:60].reshape(-1),img[20:380,295:355].reshape(-1)] 
    elif flag == 3:
        return [img[20:380,0:40].reshape(-1),img[20:380,200:240].reshape(-1),\
    img[20:380,400:440].reshape(-1)]
    elif flag == 4:
        return [img[20:380,0:30].reshape(-1),img[20:380,152:182].reshape(-1),\
    img[20:380,303:333].reshape(-1),img[20:380,455:485].reshape(-1)] 
    elif flag == 5:
        return [img[20:380,0:24].reshape(-1),img[20:380,122:146].reshape(-1),\
    img[20:380,244:268].reshape(-1),img[20:380,366:390].reshape(-1),\
    img[20:380,488:512].reshape(-1)] 





def Retrieval(query):
    # query = query[60:480,100:720]
    # query = np.squeeze(query)
    query = pre(query)
    query = np.squeeze(query)
    print(query.shape)

    flag = classify(K,query)
    
    images , crop_images = load_data(flag)
    N_ = crop_images.shape[0]
    
    query_features = get_features(flag, query)
    # store the dis
    dis = []
    # start query
    for i in range(N_):
        img = crop_images[i,:,:]
        img = np.squeeze(img)
        img_features = get_features(flag, img)
        d = 0
        for j in range(flag):
            # print(i)
            # print(j)
            # print(query.shape)
            # print(img.shape)
            # print(query_features[j].shape)
            # print(img_features[j].shape)


            d_ =  Jaccard_dis(query_features[j],img_features[j])
            d += d_[0]


        dis.append(d/flag)
    dis = np.array(dis)
    dis = np.argsort(dis)

    for idx in dis[0:6]:
        img = images[idx,:,:]
        img = np.squeeze(img)
        cv2.imshow(str(idx),img)
        cv2.waitKey(0)


query = cv2.imread('./5/quotes_2016-06-19_2016-06-24.png',cv2.IMREAD_GRAYSCALE)
Retrieval(query)