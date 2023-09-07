import streamlit as st
import tempfile
import os
from urllib.request import urlopen

st.title("Category Prediction of Images")

st.image('https://static.thestudentroom.co.uk/cms/sites/default/files/2023-04/prediction%20article.png',use_column_width=True)

uploaded_file = st.file_uploader("Upload an image from your PC", type=["jpg", "png", "jpeg"])
image_url = st.text_input("Or paste the URL of an image")

from roboflow import Roboflow

rf = Roboflow(api_key="A0HwSSkoqIrnz9hxlrBs")
project = rf.workspace().project("category")
model = project.version(1).model

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())

    st.image(temp_file.name, caption="Uploaded Image", width=300)

    jsonfile = model.predict(temp_file.name).json()
    predicted_classes = str(jsonfile["predictions"][0]["predicted_classes"][0])
    
    st.subheader("Predicted Category:")

    st.success(predicted_classes, icon="✅")

    os.unlink(temp_file.name)  # Delete the temporary file

elif image_url:
    st.image(image_url, caption="Image from URL", width=300)

    # Download the image from the URL
    with urlopen(image_url) as response:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(response.read())

    jsonfile = model.predict(temp_file.name).json()
    predicted_classes = str(jsonfile["predictions"][0]["predicted_classes"][0])
    
    st.subheader("Predicted Category:")

    st.success(predicted_classes, icon="✅")

    os.unlink(temp_file.name)  # Delete the temporary file
