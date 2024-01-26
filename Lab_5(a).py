import numpy as np
import cv2
import matplotlib.pyplot as plt

##  erosion
def erosion(img,structuring_element):
    [h,w] = img.shape
    mask = np.ones((structuring_element,structuring_element),np.uint8)
    print(mask)
    eroted_image = np.zeros_like(img)
    padding = structuring_element // 2

    for i in range(padding, h-padding):
        for j in range(padding,w-padding):
            kernel = img[i-padding :i+padding+1 , j-padding : j+padding+1]
            eroted_image[i,j] = np.min(mask * kernel)
    return eroted_image

## dilation
def dilation(img,structuring_element):
    [h,w]=img.shape
    mask = np.ones((structuring_element,structuring_element), np.uint8)
    dilated_image = np.zeros_like(img)
    padding = structuring_element // 2

    for i in range(padding, h-padding):
        for j in range(padding,w-padding):
            kernel = img[i-padding :i+padding+1 , j-padding : j+padding+1]
            dilated_image[i,j] = np.max(mask * kernel)
    return dilated_image





path = 'Digital Image Processing/Images/erosion3.png'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape

structuring_element1 = 5
eroted_image = erosion(img,structuring_element1)

structuring_element2 = 15
dilated_image = dilation(img,structuring_element2)


plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')

plt.subplot(2,2,2)
plt.imshow(eroted_image, cmap = 'gray')

plt.subplot(2,2,3)
plt.imshow(dilated_image, cmap = 'gray')










plt.show()
