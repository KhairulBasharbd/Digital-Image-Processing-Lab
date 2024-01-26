import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'Digital Image Processing/Images/aaa.jpeg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape

image_list = []
image_list.append(img.copy())
for i in range(7):
    for j in range(h):
        for k in range(w):
            img[j][k] = img[j][k] >> 1

    image_list.append(img.copy())

r,c =2,4
idx = 0
plt.figure(figsize= (12,6))
for i in range(r):
    for j in range(c):
        plt.subplot(r,c,idx+1)
        plt.imshow(image_list[idx],cmap ='gray')
        plt.title(f'{8-idx} bit')
        idx +=1


plt.tight_layout()
plt.show()        