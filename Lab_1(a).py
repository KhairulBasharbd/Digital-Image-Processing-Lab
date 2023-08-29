
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r"C:\Users\Khairul_Bashar\Desktop\Lab\Digital Image Processing\aaa.jpg", cv2.IMREAD_GRAYSCALE)



cv2.imshow("Original Image", img)
cv2.waitKey(1000)

changed_image = img.copy()
while changed_image.shape[1] >16:
    changed_image = cv2.resize(changed_image, (changed_image.shape[1] // 2, changed_image.shape[0] // 2))

    window_size = cv2.resize(changed_image, (512,512))
    cv2.imshow("Resized Image", window_size)
    key = cv2.waitKey(1000)
    

cv2.destroyAllWindows()


