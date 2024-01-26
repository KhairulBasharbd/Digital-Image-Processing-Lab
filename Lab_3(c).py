import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

## add_salt_paper_noise
def add_salt_noise(img, percentage):
    noisy_image = img.copy()
    percentage = percentage / 100
    [h,w] = img.shape
    total_pixels = h*w
    noisy_pixel = int(percentage * total_pixels)
    for i in range(noisy_pixel):
        rand_row = random.randint(0,h-1)
        rand_col = random.randint(0,w-1)
        noisy_image[rand_row,rand_col] = random.choice([0,255])
    return noisy_image
    
## harmonic_filter
def harmonic_filter(noisy_image,kernel_size):
    [h,w] = noisy_image.shape
    filtered_image = noisy_image.copy()
    padding = kernel_size //2

    for i in range(padding,h-padding):
        for j in range(padding,w-padding):
            values = []
            for x in range(i-padding , i+padding+1):
                for y in range(j - padding , j+padding+1):
                    if noisy_image[x,y] != 0:
                        values.append(1 / noisy_image[x,y])
            if values:
                filtered_image[i,j] = int (len(values)/(np.sum(values)) )

    return filtered_image      


##  geometric_filter
def geometric_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros_like(image)

    # Calculate the kernel radius
    kernel_radius = kernel_size // 2

    for i in range(kernel_radius, height - kernel_radius):
        for j in range(kernel_radius, width - kernel_radius):
            values = []
            # Iterate over the neighborhood
            for x in range(i - kernel_radius, i + kernel_radius + 1):
                for y in range(j - kernel_radius, j + kernel_radius + 1):
                    if image[x, y] != 0:
                        values.append(image[x, y])
            # Calculate the geometric mean
            if values:
                product = np.prod(values)
                filtered_image[i, j] = product ** (1 / len(values))

    return filtered_image

## add psnr
def add_psnr(img, dist_mage):

    mse = np.mean((img - dist_mage) ** 2)
    l=256
    psnr = 20 * np.log10((l-1) / np.sqrt(mse) )  
    return psnr









path = 'Digital Image Processing/Images/aaa.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape

percentage = 30
noisy_image = add_salt_noise(img, percentage)

kernel_size = 3
harmonic_filtered_Image = harmonic_filter(noisy_image,kernel_size)
harmonic_psnr = add_psnr(img, harmonic_filtered_Image)
print(harmonic_psnr)

kernel_size = 3
geometric_filtered_Image = geometric_filter(noisy_image,kernel_size)
geometric_psnr = add_psnr(img, geometric_filtered_Image)



plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')

plt.subplot(2,2,2)
plt.imshow(noisy_image, cmap = 'gray')

plt.subplot(2,2,3)
plt.title(f"harmonic PSNR {harmonic_psnr}")
plt.imshow(harmonic_filtered_Image, cmap = 'gray')

plt.subplot(2,2,4)
plt.title(f"Geometric PSNR {geometric_psnr}")
plt.imshow(geometric_filtered_Image, cmap = 'gray')

plt.tight_layout()
plt.show()