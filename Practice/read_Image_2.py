import cv2
import numpy as np
import matplotlib.pyplot as plt

#opencv reads image in BGR format
img = cv2.imread(r"C:\Users\Khairul_Bashar\Desktop\Lab\Digital Image Processing\aaa.jpg",0)

# MatplotLib reads image in RGB format
#img = plt.imread(r"C:\Users\Khairul_Bashar\Desktop\Lab\Digital Image Processing\aaa.jpg")

#Convert BGR image to RGB
#RGB_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


plt.imshow(img)
plt.waitforbuttonpress()
plt.close('all')