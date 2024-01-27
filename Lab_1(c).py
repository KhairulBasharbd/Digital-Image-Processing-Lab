import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram(img):
    arr = np.zeros(256, dtype= int)
    for i in img:
        for j in i:
            arr[j] +=1
    return arr


path = 'Digital Image Processing/myph2.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

h,w = img.shape


plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image ')


arr = histogram(img)

plt.subplot(2,2,2)
plt.bar(range(256),arr)
plt.title('Histogram of the Image')




threshold = 100
threshold_img = (img > threshold).astype(np.uint8)
plt.subplot(2,2,(3,4))
plt.imshow(threshold_img, cmap = 'gray')
plt.title(f'After thesholding the image at {threshold} ')

plt.tight_layout()
plt.show()