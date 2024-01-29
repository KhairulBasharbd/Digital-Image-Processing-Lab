
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'Digital Image Processing/Images/aaa.jpeg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (512, 512))

h,w = img.shape


min_r, max_r = 100,200
e_img = img.copy()

for i in range(h):
    for j in range(w):
        if img[i,j] > min_r & img[i,j]<max_r:
            e_img[i,j] +=50



plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')

plt.subplot(2,2,2)
plt.imshow(e_img, cmap = 'gray')
plt.title('Enhanced Image')

plt.show()


