import numpy as np
import cv2
import matplotlib.pyplot as plt

def erosion(img,structuring_element):
    [h,w] = img.shape
    mask = np.ones((structuring_element,structuring_element),np.uint8)
    eroted_image = np.zeros_like(img)
    padding = structuring_element // 2

    for i in range(padding,h-padding):
        for j in range(padding,w-padding):
            kernel = img[i-padding :i+padding+1 , j-padding : j+padding+1]
            eroted_image[i,j] = np.min(kernel * mask)
    return eroted_image




path = 'Digital Image Processing/Images/boundary.png'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape

structuring_element1 = 5
eroted_image = erosion(img,structuring_element1)

border = img - eroted_image

plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')

plt.subplot(2,2,2)
plt.imshow(border, cmap = 'gray')


plt.show()