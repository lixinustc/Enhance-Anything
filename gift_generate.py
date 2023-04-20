import cv2
import cv2
import imageio
import os
# img1=cv2.imread('./yang1.jpg',1)
# img2=cv2.imread("./yang2.jpg",1)
# img3=cv2.imread("./yang3.jpg",1)
# img1 = cv2.resize(img1, (360,460))
# img2 = cv2.resize(img2, (360,460))
# print(img1.shape[:2])
# print(img2.shape[:2])
# img3=cv2.resize(img3,(360,460))
# buff=[]
# k=31
# img = cv2.addWeighted(img1, 0.3, img3, (0.7), gamma=0)

# # cv2.imshow('img',img)
# #     # img=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2BGR)
# # cv2.imwrite("addyang.jpg", img)
# # cv2.waitKey(0)
# for i in range(k):
#     alpha=i*1/k
#     img=cv2.addWeighted(img1,alpha,img2,(1-alpha),gamma=0)

#     cv2.imshow('img',img)
#     #img=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2BGR)
#     buff.append(img)
#     cv2.waitKey(50)

import os



folder = './results_all/examples/image_2/'
imgs_list = sorted(os.listdir(folder))
buff = []
len_frame = 40
for img_n in imgs_list:
    if int(img_n.split('_0.')[-1].split(".png")[0])<len_frame:
        if int(img_n.split('_0.')[-1].split(".png")[0])%3==0:
            img = cv2.imread(os.path.join(folder, img_n))
            buff.append(img[:, :, ::-1])




gif=imageio.mimsave('yang.gif',buff,'GIF',duration=0.3)
