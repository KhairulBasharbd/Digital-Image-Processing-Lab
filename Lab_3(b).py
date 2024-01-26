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

## Find PSNR
def find_psnr(img, dist_img):
    mse = np.mean((img - dist_img) ** 2)
    psnr = 20 * np.log10 (255 / np.sqrt(mse))

    return psnr





path = 'Digital Image Processing/Images/aaa.jpeg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape



percentage = 30
noisy_image = add_salt_noise(img, percentage)

kernel_size = 3
average_filter_image1 = average_filtering(noisy_image, kernel_size)

kernel_size = 5
average_filter_image2 = average_filtering(noisy_image, kernel_size)

kernel_size = 7
average_filter_image3 = average_filtering(noisy_image, kernel_size)

average_psnr1 = find_psnr(img, average_filter_image1)
average_psnr2 = find_psnr(img, average_filter_image2)
average_psnr3 = find_psnr(img, average_filter_image3)

print(average_psnr1)
print(average_psnr2)
print(average_psnr3)



