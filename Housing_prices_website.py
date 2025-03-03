import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the pickle models
model = pickle.load(open('model(1).pkl', 'rb'))
df = pickle.load(open('houses(1).pkl', 'rb'))

html_temp = """ 
<div style = "background-color: #f06081; padding: 10px">
<h2 style = "color: white; text-align: center;">Bangalore Housing Prices Prediction
</div>
<div style = "background-color: white; padding: 5px">
<p style= "color: #7c4deb; text-align: center; font-family: Courier; font-size: 15px;">
<i>If you're considering buying a property in Bangalore, get an idea of the prices here and plan accordingly!</i></p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

image_path = 'img.jpg'
image = Image.open(image_path)
st.image(image, use_container_width=True)


# Define the features
# location
location = st.selectbox('Location', df['location'].unique())
# area_type
area_type = st.selectbox('Area Type', df['area_type'].unique())
# bhk
bhk = st.selectbox('BHK', df['bhk'].unique())
# bath
bath = st.selectbox('Bathroom', df['bath'].unique())
# balcony
balcony = st.selectbox('Balcony', df['balcony'].unique())
# total_sqft
total_sqft = st.slider('Total square feet', 300.0, 12000.0)

# Get the inputs
inputs = [[location, area_type, bhk, bath, balcony, total_sqft]]
features = pd.DataFrame(inputs, index=[0])
features.columns = ['Location', 'Area Type', 'BHK', 'Bathroom', 'Balcony', 'Total square feet']
st.markdown('##### Selected parameters')
st.write(features)

# Predict the price
def prediction():
    if (st.button('Predict Price')):
        query = np.array([location, area_type, bhk, bath, balcony, total_sqft], dtype=object)
        query = query.reshape(1,6)
        st.title(model.predict(query)[0])
prediction()

html_temp1 = """ 
<div style = "background-color: white; padding: 5px">
<p style= "color: #7c4deb; text-align: center; font-family: Courier; font-size: 15px;"><i>*Note: Price is in INR Lakhs</i></p>
</div>
<div style = "background-color: #f27eac">
<p style = "color: white; text-align: center;">Designed & Developed By: <b>Rajashri Deka</b></p>
</div>
"""
st.markdown(html_temp1, unsafe_allow_html=True)