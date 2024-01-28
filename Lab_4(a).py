
import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_gaussian_noise(img, mean=0, stddev=100):
    gaussian_noise = np.random.normal(mean, stddev, img.shape)
    noisy_image = img + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

## Butterworth low pass filter
def butterworth_low_pass_filter(image, order, cut_off_frequency):
    height, width = image.shape
    #H = np.zeros(image.shape, dtype=np.float32)
    H = np.zeros_like(image,dtype=np.float32)

    # frequncy_domain_image = np.fft.fft2(image)
    # frequncy_domain_image = np.fft.fftshift(frequncy_domain_image)
    frequency_domain_image = np.fft.fftshift(np.fft.fft2(image))
    n = order
    d0 = cut_off_frequency

    for i in range(height):
        for j in range(width):
            d = np.sqrt((i - height / 2) ** 2 + (j - width / 2) ** 2)
            H[i, j] = 1 / (1 + (d / d0) ** (2 * n))

    filteredImage = frequency_domain_image * H
    filteredImage = np.abs(np.fft.ifft2(filteredImage))
    filteredImage = filteredImage / 255
    return filteredImage

## Gaussian Low pass filter
def gaussian_low_pass_filter(image, cut_off_frequency):
    frequency_domain_image = np.fft.fftshift(np.fft.fft2(image))

    D0 = cut_off_frequency
    height, width = image.shape
    H = np.zeros(image.shape, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            d = np.sqrt((i-height/2)**2 + (j-width/2)**2)
            H[i, j] = np.exp(-(d**2) / (2*(D0)**2))

    filtered_image = frequency_domain_image * H
    filtered_image = np.abs(np.fft.ifft2(filtered_image))
    filtered_image = filtered_image / 255
    return filtered_image





path = 'Digital Image Processing/myph2.jpg'
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))

noisy_image = add_gaussian_noise(img,mean=0,stddev=40)

order = 4
cut_off_freq = 10
b_filtered_image = butterworth_low_pass_filter(noisy_image, order, cut_off_freq)
g_filtered_image = gaussian_low_pass_filter(noisy_image, cut_off_freq)




plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')

plt.subplot(2,2,2)
plt.imshow(noisy_image, cmap = 'gray')
plt.title('Gaussian noisy Image')

plt.subplot(2,2,3)
plt.imshow(b_filtered_image, cmap = 'gray')
plt.title('Butterworth low pass filtered Image')

plt.subplot(2,2,4)
plt.imshow(g_filtered_image, cmap = 'gray')
plt.title('Gaussian low pass filtered Image')

plt.tight_layout()
plt.show()