#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from urllib.parse import unquote
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.optimizers import Adam
import io 
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile , HTTPException
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
# Load the saved model
model = load_model('trained_model.h5')

app = FastAPI()


# Replace with your class labels
class_labels = {
    0: 'baseball',
    1: 'basketball',
    2: 'american football',
    3: 'golf',
    4: 'hockey',
    5: 'volleyball'
}

# Load and preprocess the data
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Assuming ResNet input size is 224x224
    img_array = img_to_array(img)
    return img_array

def preprocess_single_image(image):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    return img

def predict_single_image(image):
    image_array = preprocess_single_image(image)

    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Convert contents to an image array
        img = Image.open(io.BytesIO(contents))
        
        # Resize image to match model's input shape
        img = img.resize((224, 224))
        
        img_array = img_to_array(img)
        
        # Preprocess image array
        img_array = preprocess_input(img_array)  # Adjust this preprocessing based on your model
        
        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_label = class_labels[predicted_class_index]

        # Return the predicted class label as JSON
        return {"predicted_class": predicted_class_label}
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing image")


@app.get("/predict/{image_path}")
async def predict_get_endpoint(image_path: str):
    try:
        print(f"Received image path: {image_path}")
        
        decoded_image_path = unquote(image_path)
        print(f"Decoded image path: {decoded_image_path}")
        
        root_directory = "C:/Users/Admin/Documents/Image_classifier_ResNet/data_for_gradio/"
        image_full_path = os.path.join(root_directory, decoded_image_path)
        print(f"Full image path: {image_full_path}")
        
        if os.path.exists(image_full_path):
            predicted_class_label = predict_single_image(image_full_path)
            return {"predicted_class": predicted_class_label}
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing image")




# In[ ]:




