#Mô tả: Trong phân đoạn code này ta có 2 phần chính
#Phần 1: Dùng để chuyển từ ảnh màu thành trắng đen và nhận diện khuôn mặt người
#Phần 2: Cắt hình vào đúng nơi gương mặt nhận diện được để dễ trainning 
#Lưu ý: kích thước ảnh sau cắt phải resize thành 150x150

import cv2
import os
import numpy as np
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

#Xử lý ảnh 
for a in range(1,21):
    filename = 'E:\\\\STUDY\\\\StudyHK2\AI\\\\faceid_CNN\\\\Code\\\\test\\\\luong2\\\\luong1 ('  +str(a) + ')' + '.JPG'
    frame = cv2.imread(filename)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fa = detector.detectMultiScale(gray, 1.1, 5)
    for(x,y,w,h) in fa:
        img = cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
        path = 'E:\\\\STUDY\\\\StudyHK2\\\\AI\\\\faceid_CNN\Code\\\\testset\\\\luong\\\\luong1 ('  +str(a) + ')' + '.JPG'
        gray = gray[y:y+h,x:x+w]
        gray = cv2.resize(gray,(150,150))
        cv2.imwrite(path,gray)
        cv2.waitKey(0)
        print('1')
print('Done file')