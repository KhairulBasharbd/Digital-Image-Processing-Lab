import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'Digital Image Processing/Images/aaa.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape

plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')


last3bit_image = img & 0b11000000
plt.subplot(2,2,2)
plt.imshow(last3bit_image, cmap = 'gray')
plt.title('last 3 bit Image')


difference_image = img - last3bit_image
plt.subplot(2,2,(3,4))
plt.imshow(difference_image, cmap = 'gray')
plt.title('Difference Image')

plt.tight_layout()
plt.show()