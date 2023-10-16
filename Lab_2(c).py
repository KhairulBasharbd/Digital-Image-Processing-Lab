import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_last_three(image):

    mask = 224
    new_image = image.astype(np.uint8) & mask
    return new_image


img_path = 'Digital Image Processing/Images/aaa.jpg'
gray_image = cv2.imread(img_path,0)

new_image = make_last_three(gray_image)

plt.figure(figsize=(8, 7))
plt.subplot(121)
plt.imshow(gray_image, cmap='gray')
plt.title('Original')
plt.subplot(122)
plt.imshow(new_image, cmap='gray')
plt.title('Image of MSB-3 bits')

plt.tight_layout()
plt.show()
