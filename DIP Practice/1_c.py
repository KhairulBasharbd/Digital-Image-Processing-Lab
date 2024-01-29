import numpy as np
import cv2
import matplotlib.pyplot as plt

## Histogram
def histogram(img):
    arr = np.zeros(256, int)
    for i in img:
        for j in i:
            arr[j] +=1 
    return arr

## Thresholding
def thresholding(img, thesld):

    h,w =img.shape
    t_img = np.zeros_like(img,dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            if img[i,j] < thesld:
                t_img[i,j]=0
            else:
                t_img[i,j]=1
    return t_img


path = 'Digital Image Processing/Images/aaa.jpeg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (512, 512))

h,w = img.shape


hist_img = histogram(img)

thesld = 70

thr_img = thresholding(img, thesld)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap ='gray')
plt.title('Original Image')

plt.subplot(2,2,2)
plt.bar(range(256),hist_img)
plt.title('Histogram of Image')

plt.subplot(2,2,3)
plt.imshow(thr_img, cmap ='gray')
plt.title('Threshold Image')

plt.tight_layout()
plt.show()

