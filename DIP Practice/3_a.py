import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def add_salt_noise(img1, perc):
    img = img1.copy()
    h,w = img1.shape

    perc = perc / 100
    total_n = int(perc * h * w)

    for i in range(total_n):
        r_row = np.random.randint(0,h-1)
        r_col = np.random.randint(0,w-1)
        img [r_row, r_col] = np.random.choice([0,255])
    return img

## Average filter
def average_filtering(img,kernel):
    f_img = img.copy()
    h,w = img.shape

    padding = kernel // 2
    mask = np.ones((kernel , kernel), dtype=float) / (kernel * kernel)

    for i in range(padding, h-padding):
        for j in range(padding, w-padding):

            window = img[i-padding : i+padding+1, j-padding : j+padding+1]

            conv = window * mask
            f_img[i][j] = np.sum(conv)

    return f_img

## Median filtering
def median_filtering(noisy_image,kernel):
    f_img = noisy_image.copy()
    h,w = noisy_image.shape

    padding = kernel //2

    for i in range(padding,h-padding):
        for j in range(padding,w-padding):

            window = noisy_image[i-padding: i+padding+1, j-padding : j+padding+1]
            f_img[i,j] = np.median(window)

    return f_img

def cal_psnr(img, dist_img):
    mse = np.mean((img - dist_img) **2 )
    max = 255
    psnr = 20 * np.log10(255/np.sqrt(mse))
    return np.round(psnr, 3)


path = 'Digital Image Processing/myph2.jpg'

img = cv2.imread(path, 0)
img = cv2.resize(img,(512,512))
##print(img.shape)

percentage = 10
noisy_image = add_salt_noise(img,percentage)

kernel = 5
average_filter = average_filtering(noisy_image,kernel)
average_psnr = cal_psnr(img, average_filter)
print(average_psnr)

median_filter = median_filtering(noisy_image,kernel)
median_psnr = cal_psnr(img, median_filter)
print(median_psnr)









plt.subplot(2,2,1)
plt.imshow(img,cmap = 'gray')
plt.title("Original Image")

plt.subplot(2,2,2)
plt.imshow(noisy_image,cmap = 'gray')
plt.title("Noisy Image")

plt.subplot(2,2,3)
plt.imshow(average_filter,cmap = 'gray')
plt.title("Average filter")

plt.subplot(2,2,4)
plt.imshow(median_filter,cmap = 'gray')
plt.title("Median filter")

plt.tight_layout()
plt.show()

