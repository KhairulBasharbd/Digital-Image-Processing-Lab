import numpy as np 
import matplotlib.pyplot as plt
import cv2


img_path = 'Digital Image Processing/Images/aaa.jpg'

img = cv2.imread(img_path,0)

all_image = []
all_image.append(img)

for bits in range(7,0,-1):
    lebel = 2**bits
    normalized_image = img.astype(float)/256.0
    sample_image = np.uint8(np.floor(normalized_image * lebel))
    all_image.append(sample_image)


print(len(all_image))
row, col = 2, 4
fig, ax = plt.subplots(row, col, figsize=(9, 7))

idx = 0
for i in range(row):
    for j in range(col):
        ax[i, j].imshow(all_image[idx], cmap='gray')
        ax[i, j].set_title(f'{8 - idx} bits')
        idx += 1

plt.tight_layout()
plt.show()
