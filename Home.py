import streamlit as st
import pandas as pd
import io

st.write("# Welcome to House Price Prediction 👋")



df_origin = pd.read_csv('df_10.csv')
df_origin.drop("Unnamed: 0",axis=1,inplace=True)
st.markdown(
    """
    ### Original Dataset :

    """)
st.dataframe(df_origin.style.background_gradient(cmap='Blues'))
buffer = io.StringIO()
df_origin.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.markdown(
    """
    
    ### Description of columns:
    - Index: unique identifier
    - Title: house title
    - Description: house description (BHK is bedroom hall kitche =>no of bed rooms)
    - Amount(in rupees): total amount(Lac = one hundred thousand, Cr = 10 million) 
    - Price (in rupees): price per square foot
    - location: city
    - Carpet Area: area in sqft or other units
    - Status: house status(ready to move)
    - Floor: house floor out of total
    - Transaction: resale or new
    - Furnishing: Furnishing status() 
    - facing: direction of the house
    - overlooking: view
    - Society: district
    - Bathroom : no of bathrooms
    - Balcony : no of Balcony
    - Car Parking	: no of car parking and type
    - Ownership : Ownership type
    - Super Area: all None
    - Dimensions : all None
    - Plot Area: all None
    ### Comment on dataset:
    
    - 21 columns
    - dtypes: float64(3), int64(1), object(17)
    - most of data type is object 
    - some of these objects mixed between numbers and words ,so it needs splitting to get numbers like Description , Amount(in rupees) and Car Parking
    - some of them needs to adjust unique values
    - some columns have missing values
    - it needs a lot of cleaning 💪👌🏡
    _ the target will be price of the house 
"""
)
