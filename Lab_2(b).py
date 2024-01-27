import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'Digital Image Processing/Images/aaa.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape

c1 =1
gamma = 0.3
power_law_image = c1*np.power(img, gamma)
#power_law_image = power_law_image.astype(np.uint8)

plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(power_law_image, cmap = 'gray')
plt.title('Power law Images')


##    For 10 based logarithm
c2 = 255 / np.log10(256)
inverse_log = np.power(10,img /c2) - 1

##    For natural logarithm
# c2 = 255 / np.log(256)
# inverse_log = np.exp(img /c2) - 1

plt.subplot(2,2,2)
plt.imshow(inverse_log, cmap = 'gray')
plt.title('Inverse Log Images')



difference_image = power_law_image - inverse_log
plt.subplot(2,2,(3,4))
plt.imshow(difference_image, cmap = 'gray')
plt.title('Difference of both Images')


plt.show()