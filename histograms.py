import cv2 as cv
import numpy as np  
import matplotlib.pyplot as plt 
img=cv.imread("G:\learn opencv/DIP.jpg")
cv.imshow("original image", img)
blank=np.zeros(img.shape[:2],dtype="uint8")
#gray scale histogram
gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray scale image", gray_img)
circle=cv.circle(blank,(img.shape[0]//2,img.shape[1]//2),50,255,-1)
mask=cv.bitwise_and(gray_img,gray_img,mask=circle)
cv.imshow("masked image", mask)
gray_hist=cv.calcHist([gray_img],[0],mask,[256],[0,256]) #if you wanna show the histogram of all parts of the image,instead of mask put None
plt.figure()
plt.title("gray scale histogram")
plt.xlabel("bins")
plt.ylabel("# of pixels")
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()
cv.waitKey(0)
