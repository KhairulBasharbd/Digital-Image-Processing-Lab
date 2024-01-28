import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def add_gaussian(img,mean,stddev):
    g_noise = np.random.normal(mean,stddev,img.shape)
    noisy_image = img + g_noise
    noisy_image = np.clip(noisy_image,0,255)
    return noisy_image

def il_filter(noisy_image,cut_freq):
    d0 = cut_freq
    h,w = noisy_image.shape
    H = np.zeros_like(noisy_image,dtype=np.float32)
    f_d_img = np.fft.fftshift(np.fft.fft2(noisy_image))

    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-h/2)**2 + (j-w/2)**2) 
            #H[i,j] = d <= d0
            if d <= d0:
                H[i,j] = 1
            else:
                H[i,j] = 0
    filter_img = f_d_img * H
    filter_img = np.abs(np.fft.ifft2(filter_img))
    filter_img = filter_img / 255

    return filter_img


path = 'Digital Image Processing/myph2.jpg'

img = cv2.imread(path, 0)
img = cv2.resize(img,(512,512))
##print(img.shape)


noisy_image = add_gaussian(img, mean = 0, stddev = 20)

d01 = 5
ilpf1 = il_filter(noisy_image,d01)

d02 = 10
ilpf2 = il_filter(noisy_image,d02)

d03 = 20
ilpf3 = il_filter(noisy_image,d03)




plt.figure(figsize=(10,5))
plt.subplot(2,3,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')

plt.subplot(2,3,2)
plt.imshow(noisy_image, cmap = 'gray')
plt.title('Gaussian noisy Image')

plt.subplot(2,3,3)
plt.imshow(ilpf1, cmap = 'gray')
plt.title(f'Ideal Low pass of d0 : {d01 }')

plt.subplot(2,3,4)
plt.imshow(ilpf2, cmap = 'gray')
plt.title(f'Ideal Low pass of d0 : {d02 }')

plt.subplot(2,3,5)
plt.imshow(ilpf3, cmap = 'gray')
plt.title(f'Ideal Low pass of d0 : {d03}')

plt.tight_layout()
plt.show()