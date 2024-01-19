import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.models import load_model

import cv2
import time

from Load_EfficientNetV2B1 import load_base_model

trained_model_path = "trained_model.h5"

classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
classes_dict = {'glioma_tumor':'glioma tumor (u thần kinh đệm)', 'meningioma_tumor':'meningioma tumor (u màng não)', 'no_tumor':'no tumor (không có khối u)', 'pituitary_tumor':'pituitary tumor (u tuyến yên)'}

def get_model(trained_model_path):
    base_model = load_base_model()
    model=Sequential([
        base_model,
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(4, 4), padding='valid'),
        tf.keras.layers.Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    model=load_model(trained_model_path)
    return model

def prediction(image, model):
    img = cv2.resize(image, (512, 512))
    x_test = []
    x_test.append(img)
    x_test = np.array(x_test)
    y_pred = model.predict(x_test)
    result = classes[np.argmax(y_pred[0])]
    result = classes_dict[result]
    return result

uploaded_file = st.file_uploader("Upload ảnh MRI não bộ", type=["jpg", "png", "jpeg"])

if uploaded_file != None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # ---
    img_resize = cv2.resize(img, (250, 250)) 
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    st.image(img_rgb)
    on_click = st.button("Chẩn đoán hình ảnh")
    if on_click:
        st.write("Kết quả chẩn đoán:")
        with st.spinner('Wait for it...'):
            start_time = time.time()
            model = get_model(trained_model_path)
            result = prediction(img, model)
            end_time = time.time()
        estimated_time = end_time - start_time
        st.success(result)
        st.write(str(round(estimated_time, 2)) + 's')
