import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('C:/Users/zahid/Desktop/haarcascade_eye.xml')
img = cv2.imread('C:/Users/zahid/Desktop/DL_Projects/Mask Detectioon/Dataset/test/without mask/without mask/augmented_image_209.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0))
 
fin_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fin_img)