import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'Digital Image Processing/Images/aaa.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')

min_range, max_range = 150,200


for i in range(h):
    for j in range(w):
        if img[i,j]>min_range and img[i,j] < max_range:
            img[i,j] +=50


plt.subplot(2,2,2)
plt.imshow(img, cmap = 'gray')

plt.tight_layout()
plt.show()
