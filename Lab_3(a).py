import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def addNoise(img):
  
    row , col = img.shape
      
    number_of_pixels = random.randint(3000, 8000)
    for i in range(number_of_pixels):
     
        y_coord=random.randint(0, row - 1)  
        x_coord=random.randint(0, col - 1)  
        img[y_coord][x_coord] = 255
          
    number_of_pixels = random.randint(3000, 8000)
    for i in range(number_of_pixels):
        
        y_coord=random.randint(0, row - 1)  
        x_coord=random.randint(0, col - 1) 
        img[y_coord][x_coord] = 0    
    return img
  

img_path = 'Digital Image Processing/Images/aaa.jpg'
img = cv2.imread(img_path,0)
img_original = img.copy()

img_n = addNoise(img)
cv2.imshow('Original image', img_original)
cv2.imshow('noisy Image',img_n)
  
cv2.waitKey(0)
  
cv2.destroyAllWindows()



