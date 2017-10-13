##coding:utf8
import cv2

for i in range(201):
    img = cv2.imread('newimage\%s.jpg'%i)
    img = cv2.resize(img,(400,400))
    #image = img[70:880,70:880]
    cv2.imwrite('1\%s.jpg'%i,img)

