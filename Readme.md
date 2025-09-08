# 👗 Fashion Product Recommendation System

This is a **content-based fashion recommendation system** built with **ResNet50** and **Streamlit**.  
Users can upload a fashion product image (e.g. a dress, shoe, handbag), and the app will recommend visually similar products based on precomputed image embeddings.

---
## 🚀 Live Demo

Try the app now 👉  
🔗 **[Launch App](https://nikal25-fashion-products-recommendation-app-liu8u5.streamlit.app/)**

## 🔧 Features

- Upload an image and get top-5 visually similar fashion items.
- Built using:
  - TensorFlow + Keras (ResNet50 for feature extraction)
  - Scikit-learn (Nearest Neighbors Search)
  - Streamlit (Web Interface)

---

## 🚀 Demo

To test the application locally:

### 1. Clone the Repository

```bash
git clone https://github.com/NikaL25/fashion_products_recommendation.git
cd fashion_products_recommendation


### 2. Clone the Repository

 using Python 3.11. Install the required packages:

 pip install -r requirements.txt


### 3. Prepare Data

Make sure you have the following two files in the root of your project:

Images_features.pkl – extracted image features using ResNet50

filenames.pkl – list of image file paths corresponding to the features

⚠️ These files are essential. If not provided, you need to generate them using the original image dataset in the images/ folder. See instructions below.

## Place your .jpg images inside the /images/ folder to test or generate features.


4. Run the App

Launch the Streamlit app:

streamlit run app.py
