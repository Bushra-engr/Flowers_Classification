import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CONFIGURATION ---
MODEL_PATH = 'flowers_model.keras'  
IMG_SIZE = (128, 128)
# Alphabetical order mein rakhein (jo folder structure tha)
CLASS_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip'] 

# --- LOAD MODEL ---
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")
st.title("🌸 Flower Image Classifier")
st.write("Upload a flower photo (Daisy, Dandelion, Rose, Sunflower, or Tulip)")

try:
    model = get_model()
except Exception as e:
    st.error(f"Error: {e}. Check if {MODEL_PATH} exists.")
    st.stop()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    # --- UI: IMAGE DISPLAY (Chota size) ---
    # Hum columns use karke image ko center mein aur chota dikhayenge
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', width=250) # Fixed width for small display
    
    # --- PREPROCESSING ---
    with st.spinner('Identifying Flower...'):
        img_rgb = img.convert('RGB')
        img_resized = img_rgb.resize(IMG_SIZE)
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Sahi scaling ke liye preprocess_input zaruri hai
        prepared_img = preprocess_input(img_array.astype('float32'))
        
        # --- PREDICTION ---
        prediction = model.predict(prepared_img)
        
        # Multi-class logic: Sabse bade score ka index nikalein
        predicted_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_index]
        flower_name = CLASS_NAMES[predicted_index]

        # --- LOGIC & DISPLAY ---
        st.divider()
        if confidence > 0.4: # 40% se zyada surety ho tabhi balloons
            st.balloons()
            st.success(f"### Prediction: **{flower_name}**")
            st.write(f"**Confidence Score:** {confidence:.2%}")
            
            # Show progress bars for all classes (Optional but looks cool)
            with st.expander("See all probabilities"):
                for i, prob in enumerate(prediction[0]):
                    st.write(f"{CLASS_NAMES[i]}: {prob:.2%}")
                    st.progress(float(prob))
        else:
            st.warning("Model is not very sure. Please upload a clearer image.")

st.divider()
st.caption("Flower Classifier | Built with MobileNetV2")