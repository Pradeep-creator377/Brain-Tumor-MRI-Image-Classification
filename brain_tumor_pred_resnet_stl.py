import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

try: # Trying to load the pre trained model if it exists
    # Loading the saved Resnet50 model
    model = tf.keras.models.load_model(r"C:\Users\LAP-49\PROJECTS_GUVI\Brain Tumor MRI Image Classification\best_ResNet50.h5")
    print('Model loaded successfully!')
    model.summary() 

except Exception as e: # If loading the model fails, showing error and stopping execution
    st.error(f'Could not load model. Please check the file path. Details : {e}')
    st.stop()

# Defining the class labels corresponding to model outputs
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Streamlit app UI setup
st.title('🧠 Brain Tumor MRI Classification')
st.write('Upload a brain MRI image to predict the tumor type.')

# File uploader widget to attach files for prediction
uploaded_file = st.file_uploader('Choose an MRI image', type = ['jpg', 'jpeg', 'png'])

# If the user uploads an image  
if uploaded_file is not None:
    try:
        # Loading and preprocessing the image
        img = Image.open(uploaded_file).convert('RGB').resize((224, 224)) # Resizing the uploaded image to (224, 224) 
        img_array = np.array(img).astype(np.float32) # Converting it as a numpy array
        img_array = resnet_preprocess(img_array) # Applying Resnet50 preprocessing
        img_array = np.expand_dims(img_array, axis = 0) # Adding batch dimension

        # Running prediction using the model
        pred = model.predict(img_array)
        predicted_class = class_names[np.argmax(pred)]
        confidence_scores = pred[0] # pred[0] extracts the probabilities for the first image in the batch - [[0.05, 0.80, 0.10, 0.05]]

        # To display uploaded image with prediction result
        st.image(img, caption = f'Predicted : {predicted_class}', width = 700)
        st.success(f'Prediction Complete : {predicted_class}')

        # To show confidence scores for each class
        st.write('### Confidence Scores')
        for i, score in enumerate(confidence_scores):
            st.write(f'{class_names[i]} : {score*100 : .2f}%')

        # Plotting confidence scores as a bar chart
        fig, ax = plt.subplots()
        ax.bar(class_names, confidence_scores, color = 'skyblue')
        ax.set_ylabel('Confidence')
        ax.set_ylim(0, 1) # Range between 0 and 1
        st.pyplot(fig)

    except Exception as e:
        # Handling errors during image processing or prediction
        st.error(f'Error Processing Image. Details : {e}')

else:
    # Informing the user that an image upload is required before prediction
    st.info('Please upload an MRI image to predict')
        