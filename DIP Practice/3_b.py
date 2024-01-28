import numpy as np 
import cv2
import matplotlib.pyplot as plt 

def add_s_p_noise(img,perc):
    n_img = img.copy()
    h,w = img.shape
    perc = perc / 100
    total_n_p =int( perc * h*w)
    for i in range(total_n_p):
        r_row = np.random.randint(0,h-1)
        r_col = np.random.randint(0,w-1)
        n_img[r_row,r_col] = np.random.choice([0,255])
    return n_img


def avg_filter(img,kernel):
    f_img = img.copy()
    h,w = img.shape

    padding = kernel //2
    mask = np.ones((kernel,kernel), dtype=float ) / (kernel * kernel)
    for i in range(padding,h-padding):
        for j in range(padding,w-padding):

            window = img[i-padding:i+padding +1, j-padding : j+padding+1]
            conv = mask * window
            f_img[i,j] = np.sum(conv)

    return f_img

path = "Digital Image Processing/myph2.jpg"
img = cv2.imread(path,0)
img = cv2.resize(img,(512,512))
h,w = img.shape

perc =30
noisy_img = add_s_p_noise(img,perc)

kernel = 3
avg1 = avg_filter(noisy_img,kernel)

kernel = 5
avg2 = avg_filter(noisy_img,kernel)

kernel = 7
avg3 = avg_filter(noisy_img,kernel)

plt.figure(figsize=(10,5))
plt.subplot(2,3,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')


plt.subplot(2,3,2)
plt.imshow(noisy_img, cmap = 'gray')
plt.title('Noisy Image')


plt.subplot(2,3,3)
plt.imshow(avg1, cmap = 'gray')
plt.title('Average filtered Image 3x3')

plt.subplot(2,3,4)
plt.imshow(avg2, cmap = 'gray')
plt.title('Average filtered Image 5x5')

plt.subplot(2,3,5)
plt.imshow(avg3, cmap = 'gray')
plt.title('Average filtered Image 7x7')

plt.tight_layout()
plt.show()