
import numpy as np
import cv2
import matplotlib.pyplot as plt


path = 'Digital Image Processing/myph2.jpg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (512, 512))

h,w = img.shape

msb_img = img & 0b11100000



plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')

plt.subplot(2,2,2)
plt.imshow(msb_img, cmap = 'gray')
plt.title('Msb  Image')

difference_image = img - msb_img
plt.subplot(2,2,(3,4))
plt.imshow(difference_image, cmap = 'gray')
plt.title('Difference Image')

plt.show()