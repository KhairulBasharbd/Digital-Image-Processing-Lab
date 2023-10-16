import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_generate(image):
    pixel_counts = np.zeros(256, dtype=int)
    for row in image:
        for pixel_value in row:
            pixel_counts[pixel_value] += 1

    return pixel_counts

img_path = 'Digital Image Processing/Images/aaa.jpg'
img = cv2.imread(img_path,0)

pixel_counts = histogram_generate(img)


threshold = 90
segmented_image = (img > threshold).astype(np.uint8)*255

#print(segmented_image)

plt.figure(figsize=(8, 7))
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(2,2,2)
plt.bar(range(256), pixel_counts)
plt.title('Histogram')
plt.subplot(2,2,(3,4))
plt.imshow(segmented_image, cmap='gray')
plt.title(f'binary Threshold:{threshold}')

plt.tight_layout()
plt.show()



