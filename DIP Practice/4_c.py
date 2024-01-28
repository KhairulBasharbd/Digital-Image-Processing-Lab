import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def add_gaussian(img,mean,stddev):
    g_noise = np.random.normal(mean,stddev,img.shape)
    noisy_image = img + g_noise
    noisy_image = np.clip(noisy_image,0,255)
    return noisy_image

## Ideal High pass
def ih_filter(noisy_image,cut_freq):
    d0 = cut_freq
    h,w = noisy_image.shape
    H = np.zeros_like(noisy_image,dtype=np.float32)
    f_d_img = np.fft.fftshift(np.fft.fft2(noisy_image))

    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-h/2)**2 + (j-w/2)**2) 
            #H[i,j] = d <= d0
            if d < d0:
                H[i,j] = 0
            else:
                H[i,j] = 1
    filter_img = f_d_img * H
    filter_img = np.abs(np.fft.ifft2(filter_img))
    filter_img = filter_img / 255

    return filter_img


## Gaussian high pass filter
def gh_filter(noisy_image,cut_freq):
    d0 = cut_freq
    h,w = noisy_image.shape
    H = np.zeros_like(noisy_image,dtype=np.float32)
    f_d_img = np.fft.fftshift(np.fft.fft2(noisy_image))

    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-h/2)**2 + (j-w/2)**2) 
            
            H[i,j] = 1 - np.exp(-(d**2 / (2 * d0**2))) 

    filter_img = f_d_img * H
    filter_img = np.abs(np.fft.ifft2(filter_img))
    filter_img = filter_img / 255

    return filter_img





path = 'Digital Image Processing/myph2.jpg'

img = cv2.imread(path, 0)
img = cv2.resize(img,(512,512))
##print(img.shape)


noisy_image = add_gaussian(img, mean = 0, stddev = 20)

d0 = 5
ihpf = ih_filter(noisy_image,d0)

ghpf = gh_filter(noisy_image,d0)


plt.figure(figsize=(10,5))
plt.subplot(2,3,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')

plt.subplot(2,3,2)
plt.imshow(noisy_image, cmap = 'gray')
plt.title('Gaussian noisy Image')

plt.subplot(2,3,3)
plt.imshow(ihpf, cmap = 'gray')
plt.title(f'Ideal High pass of d0 : {d0}')

plt.subplot(2,3,4)
plt.imshow(ghpf, cmap = 'gray')
plt.title(f'Gaussian High pass of d0 : {d0}')

plt.tight_layout()
plt.show()