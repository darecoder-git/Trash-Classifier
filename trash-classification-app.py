import streamlit as st
import base64
import sklearn
import numpy as np
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
scal=MinMaxScaler()
import pickle as pkl
pkl.dump(model,open("final_model.p","wb"))
model.save("final_model.p")
#Load the saved model
model=pkl.load(open("final_model.p","rb"))
from PIL import Image

# Load your classification model


# Create a function that uses the model to predict the classification
# of an image of garbage
def predict(image):
    prediction = model.predict(image)
    return prediction

# Create the main function of your Streamlit app
def main():
    st.title("Garbage Classification App")
    st.markdown("Upload an image of garbage to classify it")

    # Allow the user to upload an image
    image = st.file_uploader("Choose an image", type=["jpg", "png"])

    # Check if the user has uploaded an image
    if image is not None:
        # Convert the uploaded image to a PIL image
        image = Image.open(image)

        # Use the classification model to predict the classification
        # of the image
        prediction = predict(image)

        # Display the prediction to the user
        st.image(image)
        st.markdown(f"This is a {prediction}")

# Run the main function
if __name__ == "__main__":
    main()


