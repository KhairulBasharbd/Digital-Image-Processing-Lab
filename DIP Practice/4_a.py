import numpy as np 
import matplotlib.pyplot as plt 
import cv2


def add_gaussian(img, mean = 0, stddev = 50):
    g_noise = np.random.normal(mean, stddev, img.shape)
    n_img = img + g_noise
    #n_img = np.clip(n_img,0,255)
    return n_img.astype(np.uint8)

## Butterworth filtering
def b_filtering(noisy_image,order,cut_freq):
    h,w = noisy_image.shape
    n = order
    d0 = cut_freq

    H = np.zeros_like(noisy_image,dtype=np.float32)
    freq_d_img = np.fft.fftshift(np.fft.fft2(noisy_image))

    for i in range(h):
        for j in range(w):
            d = np.sqrt( (i - (h/2))**2 + (j-(w/2)) **2)

            H[i,j] = (1 / (1 + (d/d0) **(2*n)))
    
    filtered_img = freq_d_img * H
    filtered_img = np.abs(np.fft.ifft2(filtered_img))
    filtered_img = filtered_img / 255

    return filtered_img


## Gaussian filtering
def g_filtering(noisy_image,cut_freq):
    h,w = noisy_image.shape
    d0 = cut_freq

    H = np.zeros_like(noisy_image,dtype=np.float32)
    f_d_img = np.fft.fftshift(np.fft.fft2(noisy_image))

    for i in range(h):
        for j in range(w):

            d = np.sqrt((i-(h/2))** 2 + (j - (w/2)) ** 2)
            H[i,j] = np.exp(- (d **2 ) / (2* d0**2))
    filter_img = f_d_img * H
    filter_img = np.abs(np.fft.ifft2(filter_img))
    filter_img = filter_img /255
    return filter_img




path = 'Digital Image Processing/myph2.jpg'

img = cv2.imread(path, 0)
img = cv2.resize(img,(512,512))
##print(img.shape)


noisy_image = add_gaussian(img, mean = 0, stddev = 20)

order = 4
cut_freq = 10
b_filter = b_filtering(noisy_image,order,cut_freq)

g_filter = g_filtering(noisy_image,cut_freq)


plt.subplot(2,2,1)
plt.imshow(img,cmap = 'gray')
plt.title("Original Image")

plt.subplot(2,2,2)
plt.imshow(noisy_image,cmap = 'gray')
plt.title("Noisy Image")

plt.subplot(2,2,3)
plt.imshow(b_filter,cmap = 'gray')
plt.title("Butter worth low filter Image")

plt.subplot(2,2,4)
plt.imshow(g_filter,cmap = 'gray')
plt.title("Gaussian low pass filter")

plt.show()

