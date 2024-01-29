
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'Digital Image Processing/Images/aaa.jpeg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (512, 512))

h,w = img.shape
f=2
imglist = []
#imglist.append(img)

for i in range(8):
    temp_img = np.zeros((h//f,w//f), dtype=np.uint8)

    if temp_img.shape[0] >0 :
        for j in range(0,h,f):
            for k in range(0,w,f):
                temp_img[j//f,k//f] = img[j,k]
        imglist.append(temp_img)
    f = f*2


r,c = 2,4
plt.figure(figsize=(12,8))
idx = 0
for i in range(r):
    for j in range(c):
        plt.subplot(r,c,idx+1)
        plt.imshow(imglist[idx], cmap = 'gray')
        m,n = imglist[idx].shape
        plt.title(f'{m}x{n} image')
        idx +=1
plt.tight_layout()
plt.show()
