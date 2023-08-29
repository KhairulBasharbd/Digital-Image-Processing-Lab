import cv2
img = cv2.imread(r"C:\Users\Khairul_Bashar\Desktop\Lab\Digital Image Processing\aa.jpeg", cv2.IMREAD_REDUCED_COLOR_2)

cv2.imshow("image", img)
 
cv2.waitKey(0)
 
cv2.destroyAllWindows()