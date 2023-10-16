import cv2
import numpy as np

# Load the grayscale image
gray_image = cv2.imread(r"C:\Users\Khairul_Bashar\Desktop\Lab\Digital Image Processing\aaa.jpg", cv2.IMREAD_GRAYSCALE)

# Convert it to an RGB image
rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# Save the RGB image
#cv2.imwrite('rgb_image.jpg', rgb_image)
cv2.imshow("Resized Image", rgb_image)
key = cv2.waitKey()
cv2.destroyAllWindows()
