import pandas as pd 
import numpy as np 
import pickle as pk 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import streamlit as st

model = pk.load(open('LRModel.pkl','rb'))

st.header('Car Price Dekho')

cars = pd.read_csv('Cleaned_Car_data.csv')



car_brands = sorted(cars['company'].unique())
placeholder_brand = ""
car_brands.insert(0, placeholder_brand)

company = st.selectbox('Select Car Brand', car_brands)


# Initialize selected model as empty initially
selected_model = ""

# Display select box for car models
selected_brand_models = cars['name'].tolist()
selected_brand_models.insert(0, "")  # Insert blank option

# Filter car models based on selected brand
if company != placeholder_brand:
    selected_brand_models = cars[cars['company'] == company]['name'].tolist()
    selected_brand_models.insert(0, "")  # Insert blank option

# Display select box for car models related to the selected brand
name = st.selectbox('Select Car Model', selected_brand_models)

year = st.text_input('Year', placeholder='Enter the manufacturing year')
kms_driven = st.text_input('Kilometers Travelled', placeholder='Enter the kilometers travelled (Eg: 45000)')

# Get unique fuel types and sort them
fuel_types = sorted(cars['fuel_type'].unique())

# Display select box for fuel types with a placeholder
fuel_placeholder = ""
fuel = st.selectbox('Fuel type', [fuel_placeholder] + fuel_types)

# Handle selection of placeholder
if fuel == fuel_placeholder:
    fuel = None

# Remove placeholder if it's selected
if fuel == fuel_placeholder:
    fuel = None

if st.button("Predict"):

    input_data_model = pd.DataFrame(
    [[company,name, year, kms_driven, fuel]],
    columns=['company','name','year','kms_driven','fuel_type'])
    #Predict car price
    car_price = model.predict(input_data_model)

    #Display prediction result
    st.success('Predicted Car Price: ₹ {:.2f}'.format(car_price[0]))

