import streamlit as st
import requests
import numpy as np
import pandas as pd
import tensorflow as tf

st.title('마스크 착용 감지 모델')


# requests.get('https://github.com/imngooh/mini_streamlit/raw/main/acc_0.943_vgg19.h5')


model = tf.keras.models.load_model('acc_0.943_vgg19.h5')
model.summary()
tf.keras.utils.plot_model(model)