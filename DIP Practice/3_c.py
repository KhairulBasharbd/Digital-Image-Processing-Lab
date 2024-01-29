import numpy as np
import matplotlib.pyplot as plt
import cv2

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

## Cal.. PSNR
def cal_psnr(img, dist_img):
    mse  = np.mean((img - dist_img) **2)
    max =255
    psnr = 20 * np.log10(max / np.sqrt(mse))
    return psnr

## Geometric Filter
def geo_filtering(noisy_image,kernel):
    f_img = noisy_image.copy()
    h,w = noisy_image.shape

    padding = kernel // 2

    for i in range(padding, h-padding):
        for j in range(padding,w-padding):
            window = noisy_image[i-padding : i+padding+1, j-padding : j+padding+1]
            window = window + 0.1

            product = np.prod(window)
            f_img[i,j] = product ** (1/(kernel **2))
    return f_img




## Harmonic filter
def har_filtering(noisy_image,kernel):
    f_img = noisy_image.copy()
    h,w = noisy_image.shape

    padding = kernel // 2

    for i in range(padding, h-padding):
        for j in range(padding,w-padding):
            #window = noisy_image[i-padding : i+padding+1, j-padding : j+padding+1]
            values = []
            for x in range(i-padding , i+padding+1):
                for y in range(j-padding , j+padding+1):
                    if noisy_image[x,y] != 0:
                        values.append(1/(noisy_image[x,y]))
            if values:
                f_img[i,j] =int (len(values))/ (np.sum(values))



    return f_img




path = 'Digital Image Processing/myph2.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

[h,w] = img.shape

## percentage of noise
percentage = 10
noisy_image = add_salt_noise(img, percentage)

kernel = 5
har_filter = har_filtering(noisy_image,kernel)
geo_filter = geo_filtering(noisy_image,kernel)



plt.figure(figsize=(10,7))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')


plt.subplot(2,2,2)
plt.imshow(noisy_image, cmap = 'gray')
plt.title('Noisy Image')


plt.subplot(2,2,3)
plt.imshow(geo_filter, cmap = 'gray')
plt.title('geometric filtered Image')

plt.subplot(2,2,4)
plt.imshow(har_filter, cmap = 'gray')
plt.title('Harmonic filtered Image')


plt.show()