import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st

st.set_page_config(page_title="Fashion Recommendation System")
st.header('👗Fashion Recommendation System')
st.markdown("Upload a fashion product image and get similar recommendations!")

image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))


def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result


base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.models.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

neighbors = NearestNeighbors(
    n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(image_features)

upload_file = st.file_uploader("📤 Upload Image", type=['jpg', 'jpeg', 'png'])

if upload_file is not None:
    os.makedirs('upload', exist_ok=True)
    saved_path = os.path.join('upload', upload_file.name)

    with open(saved_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    st.subheader('🖼 Uploaded Image')
    st.image(upload_file)

    input_img_features = extract_features_from_images(saved_path, model)
    distances, indices = neighbors.kneighbors([input_img_features])

    st.subheader('🔍 Recommended Similar Images')
    cols = st.columns(5)

    for i, col in enumerate(cols):
        if i + 1 < len(indices[0]):
            col.image(filenames[indices[0][i + 1]], caption=f"Similar {i+1}")
