
import numpy as np
import cv2
import matplotlib.pyplot as plt

def power_law(img,c,gm):
    p_img = np.zeros_like(img)
    h,w =img.shape

    p_img = c * np.power(img, gm)
    return p_img

    



path = 'Digital Image Processing/myph2.jpg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (512, 512))

h,w = img.shape

c,gm = 1.0, 0.3 
pl_t = power_law(img,c,gm)

c1 = (255) / (np.log10(256))
il_t = np.exp(img / c1) -1



plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')


plt.subplot(2,2,2)
plt.imshow(pl_t, cmap = 'gray')
plt.title('Power trensform Image')

plt.subplot(2,2,3)
plt.imshow(il_t, cmap = 'gray')
plt.title('Inverse log trensform Image')

difference_image = il_t - pl_t
plt.subplot(2,2,4)
plt.imshow(difference_image, cmap = 'gray')
plt.title('Difference of both Images')


plt.tight_layout()
plt.show()