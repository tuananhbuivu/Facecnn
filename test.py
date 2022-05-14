#Test kiểm tra mô hình train
from tensorflow.keras.models import load_model
model=load_model('E:\\\\face_cnn\\\\classifier_faceCNN.h5')
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
test_img=load_img('E:\\\\face_cnn\\\\tuan (1).jpg',target_size=(150,150))
plt.imshow(test_img)
#plt.show(test_img)
import numpy as np
test_img= img_to_array(test_img)
test_img=test_img/255
test_img=np.expand_dims(test_img,axis=0)
result=model.predict(test_img)
if round(result[0][0])==1:
   prediction="LE QUOC AN"
elif round(result[0][1])==1:
   prediction="NGUYEN HAO ANH"
elif round(result[0][2])==1:
   prediction="BUI VU TUAN ANH"  
print('=====> KET QUA: ' + prediction)
