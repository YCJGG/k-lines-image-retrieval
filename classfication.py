
import numpy as np

def classify(K,crop_img):

    #flag = -1

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
    l5_1 =  np.sum(crop_img[:,10:15])
    l5_2 =  np.sum(crop_img[:,132:137])
    l5_3 =  np.sum(crop_img[:,254:259])
    l5_4 =  np.sum(crop_img[:,376:381])
    l5_5 =  np.sum(crop_img[:,498:503])
    # 4 lines
    l4_1 =  np.sum(crop_img[:,13:18])
    l4_2 =  np.sum(crop_img[:,165:170])
    l4_3 =  np.sum(crop_img[:,316:321])
    l4_4 =  np.sum(crop_img[:,468:473])
    # 3 lines
    l3_1 =  np.sum(crop_img[:,18:23])
    l3_2 =  np.sum(crop_img[:,218:223])
    l3_3 =  np.sum(crop_img[:,418:423])
    #2 lines
    l2_1 =  np.sum(crop_img[:,28:33])
    l2_2 =  np.sum(crop_img[:,323:328])
    # 1 lines 
    l_1 = np.sum(crop_img[:,53:58])
    #cv2.imshow('ori',img)
    #cv2.waitKey(0)

    #print(l_1)
    #classfication
    if (l5_1>K) and (l5_2>K) and (l5_3>K) and (l5_4>K) and (l5_5>K):
        # c_l5.append(crop_img)
        # l5.append(img)  
        # cv2.imwrite('./5/'+cont,img)
        flag = 5

    elif (l4_1>K) and (l4_2>K) and (l4_3>K) and (l4_4>K):
        # c_l4.append(crop_img)
        # l4.append(img)  
        # cv2.imwrite('./4/'+cont,img)
        flag = 4

    elif (l3_1>K) and (l3_2>K) and (l3_3>K) :
        # c_l3.append(crop_img)
        # l3.append(img)  
        # cv2.imwrite('./3/'+cont,img)
        flag = 3

    elif (l2_1>K) and (l2_2>K):
        # c_l2.append(crop_img)
        # l2.append(img)  
        # cv2.imwrite('./2/'+cont,img)
        flag = 2

    elif (l_1>K):
        # c_l1.append(crop_img)
        # l1.append(img)  
        # cv2.imwrite('./1/'+cont,img)

        flag = 1
    return flag