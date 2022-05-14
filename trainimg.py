#Mô tả: Trong phân đoạn code này ta có 2 phần chính
#Phần 1: Dùng để chuyển từ ảnh màu thành trắng đen và nhận diện khuôn mặt người
#Phần 2: Cắt hình vào đúng nơi gương mặt nhận diện được để dễ trainning 
#Lưu ý: kích thước ảnh sau cắt phải resize thành 150x150
import cv2
import os
import numpy as np
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#Xử lý ảnh 
for a in range(1,61):
    filename = 'E:\\\\face_cnn\\\\training\\\\tuan\\\\tuan ('  +str(a) + ')' + '.jpg'
    frame = cv2.imread(filename)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fa = detector.detectMultiScale(gray, 1.1, 5)
    for(x,y,w,h) in fa:
        img = cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
        path = 'E:\\\\face_cnn\\\\training\\\\tuan1\\\\tuan ('  +str(a) + ')' +'.jpg'
        gray = gray[y:y+h,x:x+w]
        gray = cv2.resize(gray,(150,150))
        cv2.imwrite(path,gray)
        cv2.waitKey(0)
    print(a)
print('Done file')