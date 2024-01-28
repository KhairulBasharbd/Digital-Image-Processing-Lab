import numpy as np
import cv2
import matplotlib.pyplot as plt

## Erosion
def erosion(img, ste):
    
    h,w = img.shape
    SE = np.ones((ste,ste),np.uint8)
    e_img = np.zeros_like(img)
    padding = ste //2

    for i in range(padding,h-padding):
        for j in range(padding, w-padding):

            window = img[i-padding : i+padding+1, j-padding:j-padding+1]
            e_img [i,j] = np.min(window * SE)
    return e_img

## Dialation
def dialation(img,se1):
    h,w = img.shape
    d_img = np.zeros_like(img)
    SE = np.ones((se1,se1),np.uint8)
    padding = se1 //2

    for i in range(padding,h-padding):
        for j in range(padding,w-padding):
            window = img[i-padding:i+padding+1, j-padding:j+padding+1]
            d_img[i,j] = np.max(window * SE)
    return d_img

path = 'Digital Image Processing/Images/erosion.jpg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (256,256))

se = 15
er1 = erosion(img,se)

se1 = 15
dl1 = dialation(er1,se1)

plt.figure(figsize=(10,8))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2,2 )
plt.imshow(er1, cmap='gray')
plt.title('Eroted Image')

plt.subplot(2, 2, 3 )
plt.imshow(dl1, cmap='gray')
plt.title('Dialated Image')

plt.show()