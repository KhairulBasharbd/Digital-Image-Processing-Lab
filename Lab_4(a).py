
import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_gaussian_noise(img, mean=0, stddev=1):
    gaussian_noise = np.random.normal(mean, stddev, img.shape)
    noisy_image = img + gaussian_noise
    #noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


path = 'Digital Image Processing/Images/aaa.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

gaussian_noise = add_gaussian_noise(img,mean=0,stddev=40)


plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')

plt.subplot(2,2,2)
plt.imshow(gaussian_noise, cmap = 'gray')







plt.show()