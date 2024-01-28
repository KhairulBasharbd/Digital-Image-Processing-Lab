import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_salt_noise(img1, perc):
    img = img1.copy()
    h, w = img1.shape

    perc = perc / 100
    total_n = int(perc * h * w)

    for i in range(total_n):
        r_row = np.random.randint(0, h-1)
        r_col = np.random.randint(0, w-1)
        img[r_row, r_col] = np.random.choice([0, 255])

    return img

def geometric_mean_filtering(img, kernel_size):
    h, w = img.shape
    padding = kernel_size // 2
    filtered_img = np.zeros((h, w), dtype=np.float64)

    for i in range(padding, h-padding):
        for j in range(padding, w-padding):
            window = img[i-padding: i+padding+1, j-padding: j+padding+1]
            window = window + 0.01
            product = np.prod(window)
            filtered_img[i, j] = product**(1/(kernel_size**2))

    return np.uint8(filtered_img)

path = 'Digital Image Processing/Images/aaa.jpeg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (512, 512))

percentage = 10
noisy_image = add_salt_noise(img, percentage)

kernel_size = 5
geometric_filtered = geometric_mean_filtering(noisy_image, kernel_size)

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.subplot(2, 2, 3)
plt.imshow(geometric_filtered, cmap='gray')
plt.title('Geometric Mean Filtered Image')

plt.show()
