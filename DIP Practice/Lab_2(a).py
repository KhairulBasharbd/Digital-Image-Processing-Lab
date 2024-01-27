import cv2
import numpy as np
import matplotlib.pyplot as plt

def brightness_enhancement(image, enhancement_factor, min_intencity, max_intencity):
    enhanced_image = np.copy(image)

    for y in range(enhanced_image.shape[0]):
        for x in range(enhanced_image.shape[1]):
            gray_value = enhanced_image[y, x]
            if gray_value >= min_intencity and gray_value <= max_intencity:
                new_gray_value = gray_value + enhancement_factor
                if new_gray_value > 255:
                    new_gray_value = 255
                elif new_gray_value < 0:
                    new_gray_value = 0
                enhanced_image[y, x] = new_gray_value

    return enhanced_image


img_path = 'Digital Image Processing/Images/aaa.jpg'
gray_image = cv2.imread(img_path,0)

factor, low, high = 50, 150, 205
enhanced_image = brightness_enhancement(gray_image, factor, low, high)


fig, ax = plt.subplots(1,2, figsize=(8, 7))

ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(enhanced_image, cmap='gray')
ax[1].set_title(f'Enhanced between[{low}-{high}] by {factor}')

plt.tight_layout()
plt.show()


