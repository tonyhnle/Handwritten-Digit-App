"""
Created on Sat Jan  8 00:26:27 2022

@author: hoain
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2  
import tensorflow as tf
from tensorflow.keras.models import load_model
st.title('Handwritten Digits Recognition')

st.write("""This deep-learning app is an implementation of handwritten digit recognition. 
         This is implemented by training the MNIST dataset containig 60,000 training images
         of handwritten digits from zero to nine and 10,000 images for testing.
         The handwritten digits images are represented as a 28×28 matrix where each cell 
         contains grayscale pixel value. To train this data, we are using TensorFlow to create a
         Convolutional Neural Network (CNN), which is a type of Deep Learning Algorithm that takes an image
         as an input and learns the various features of the image through filters, which allows
         them to learn the important features specific to an image, allowing the model to differentiate
         inputted images. In this case, we are working with digits.
         More on CNNs at https://data-flair.training/blogs/convolutional-neural-networks-tutorial/""")
         
st.subheader('Draw Digit on Canvas')

SIZE = 192
model = load_model('my_model.h5')

#creating canvas component

canvas_result = st_canvas(
    fill_color = "#ffffff",
    stroke_width = 10,
    stroke_color = '#ffffff',
    background_color = '#000000',
    height=300,width=300,
    drawing_mode = 'freedraw',
    key = "canvas")

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28,28))
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation = cv2.INTER_NEAREST)
    st.write('Input Image (rescaled for prediction):')
    st.image(img_rescaling)
    
if st.button('Predict Digit'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model.predict(test_x.reshape(1,28,28))
    st.write("The predicted digit is " , np.argmax(pred[0]))
    st.bar_chart(pred[0])
    
st.write("Streamlit, Streamlit Drawable Canvas, NumPy, OpenCV, and TensorFlow were used for this app.")