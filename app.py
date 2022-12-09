import streamlit as st
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# 마스크 착용 감지 모델
# 기능 : 이용자로부터 얼굴 이미지를 입력받아, 모델을 통해 마스크를 썼는지 안 썼는지 감지 후 결과 출력


# 제목
st.title('마스크 착용 감지 모델')

# 모델 임포트
model = tf.keras.models.load_model('acc_0.943_vgg19.h5')
model.summary()

# 사진 입력받기
uploaded_file = st.file_uploader("얼굴 사진을 올려주세요!")

# 예측 및 결과 출력
height = 150
width = 150

def title_predict(path) :
    img = tf.keras.preprocessing.image.load_img(path, target_size=(height,width))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    pred = model.predict(np.array([img]))

    if pred[0][0] > 0.5 : 
       return f'Without Mask : {pred[0][0]*100 : 0.2f}%', 0
    else : 
        return f'With Mask : {(1-pred[0][0])*100 : 0.2f}%', 1

if uploaded_file is not None:
    if title_predict(uploaded_file)[1] == 1:
        st.write('## 😷마스크를 착용하셨군요!')
    else:
        st.write('## 🙂마스크를 착용하지 않으셨군요!')
        
    img = plt.imread(uploaded_file)
    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(img)
    ax.set_title(title_predict(uploaded_file)[0])
    st.pyplot(fig)
    
