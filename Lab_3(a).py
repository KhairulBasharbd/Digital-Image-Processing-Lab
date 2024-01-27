import numpy as np
import cv2
import matplotlib.pyplot as plt
#import random

## add_salt_paper_noise
def add_salt_noise(img, percentage):
    noisy_image = img.copy()
    percentage = percentage / 100
    [h,w] = img.shape
    total_pixels = h*w
    noisy_pixel = int(percentage * total_pixels)
    
    for i in range(noisy_pixel):
        rand_row = np.random.randint(0,h-1)
        rand_col = np.random.randint(0,w-1)
        noisy_image[rand_row,rand_col] = np.random.choice([0,255])
    return noisy_image

## perform average filtering
def average_filtering(noisy_image,kernel_size):
    [h,w] = noisy_image.shape
    filtered_image = noisy_image.copy()
    padding = kernel_size // 2
    mask = (np.ones((kernel_size , kernel_size), dtype=float) ) / (kernel_size ** 2)

    for i in range(padding, h - padding):
        for j in range(padding, w - padding):
            window = noisy_image[i - padding : i + padding + 1 , j - padding : j+ padding +1]
            conv_res = mask * window
            filtered_image[i , j] = np.sum(conv_res)

    return filtered_image

## perform median filtering
def median_filtering(noisy_image, kernel_size):
    [h,w] = noisy_image.shape
    filtered_image = noisy_image.copy()
    padding = kernel_size // 2

    for i in range(padding,h-padding):
        for j in range(padding, w - padding):
            window = noisy_image[i - padding : i+padding+1 , j -padding : j + padding+1]
            filtered_image[i , j] = np.median(window)

    return filtered_image


## Find PSNR
def find_psnr(img, dist_img):
    mse = np.mean((img - dist_img) ** 2)
    psnr = 20 * np.log10 (255 / np.sqrt(mse))

    return psnr





path = 'Digital Image Processing/myph2.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape

## percentage of noise
percentage = 50
noisy_image = add_salt_noise(img, percentage)

kernel_size = 5
average_filter_image = average_filtering(noisy_image, kernel_size)
median_filter_image = median_filtering(noisy_image, kernel_size)

average_psnr = find_psnr(img, average_filter_image)
median_psnr = find_psnr(img, median_filter_image)

print(f'PSNR of Average filter : {average_psnr} ')
print(f'PSNR of Median filter : {median_psnr} ')


plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')


plt.subplot(2,2,2)
plt.imshow(noisy_image, cmap = 'gray')
plt.title('Noisy Image')


plt.subplot(2,2,3)
plt.imshow(average_filter_image, cmap = 'gray')
plt.title('Average filtered Image')


plt.subplot(2,2,4)
plt.imshow(median_filter_image, cmap = 'gray')
plt.title('Median filtered Image')

plt.tight_layout()
plt.show()