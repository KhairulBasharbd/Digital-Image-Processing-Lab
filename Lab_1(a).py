import numpy as np
import cv2
import matplotlib.pyplot as plt

path = 'Digital Image Processing/myph2.jpg'
img = cv2.imread(path,0)
print(img.shape)

img = cv2.resize(img,(512,512))
h,w = img.shape

f=2
image_list = []

for i in range(8):
    temp_img = np.zeros((h//f,w//f), dtype = np.uint8)
    
    if temp_img.shape[0] >0 and temp_img.shape[1] >0 :
        for j in range(0,h,f):
            for k in range(0,w,f):
                temp_img[j//f][k//f] = img[j][k]

            
        image_list.append(temp_img)

    f = f*2

r,c = 2,4
fig, ax = plt.subplots(r,c,figsize = (12,6))
idx=0
for i in range(r):
    for j in range(c):

        ax[i, j].imshow(image_list[idx],cmap = 'gray')
        [m,n] = image_list[idx].shape
        ax[i, j].set_title(f'{m}x{n}')
        idx +=1

plt.tight_layout() 
plt.show()
