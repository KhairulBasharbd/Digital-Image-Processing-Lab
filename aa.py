import numpy as np
import matplotlib.pyplot as plt
import cv2

img_path = 'Digital Image Processing/Images/aa.jpeg'

img = cv2.imread(img_path, 0)

all_images = []
all_images.append(img)

for bits in range(7, 0, -1):
    level = 2**bits
    normalized_image = img.astype(float) / 255.0  # Normalize pixel values to [0, 1]
    sample_image = np.uint8(np.floor(normalized_image * (level)))  # Quantize to the specified bit depth
    all_images.append(sample_image)

row, col = 2, 4
fig, ax = plt.subplots(row, col, figsize=(12, 6))

idx = 0
for i in range(row):
    for j in range(col):
        ax[i, j].imshow(all_images[idx], cmap='gray')
        ax[i, j].set_title(f'{8 - idx} bits')
        idx += 1

plt.tight_layout()
plt.show()
