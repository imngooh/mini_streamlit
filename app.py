import streamlit as st
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


st.title('마스크 착용 감지 모델')


# requests.get('https://github.com/imngooh/mini_streamlit/raw/main/acc_0.943_vgg19.h5')


model = tf.keras.models.load_model('acc_0.943_vgg19.h5')
model.summary()
# tf.keras.utils.plot_model(model)

height = 150
width = 150

def title_predict(path) :
    img = tf.keras.preprocessing.image.load_img(path, target_size=(height,width))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    pred = model.predict(np.array([img]))

    if pred[0][0] > 0.5 : 
       return f'Without Mask : {pred[0][0]*100 : 0.2f}%'
    else : 
        return f'With Mask : {(1-pred[0][0])*100 : 0.2f}%'


img = plt.imread('with_mask.png')
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title(title_predict('with_mask.png'))
st.pyplot(fig)
