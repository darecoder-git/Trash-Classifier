import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from tensorflow.python.keras.models import load_model


MODEL_PATH='./model_model.h5'
#load the model
model = load_model(MODEL_PATH)
model.summary()



'''
#the model is loaded
st.write("""
         # Trash Classification
         """
         )
st.write("This is a simple image classification web app to predict trash in an image")
#upload the image
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#do predict the image'





def import_and_predict(image_data, model):
    size = (300,300)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    string = "This image most likely belongs to {} with a {:.2f} percent confidence."
    st.write(string.format(class_names[np.argmax(predictions)], 100 * np.max(predictions)))
    st.run(debug=True)

'''