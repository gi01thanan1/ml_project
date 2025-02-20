import streamlit as st
import numpy as np
import pandas as pd


# from datetime import datetime, timedelta
# from geopy.distance import great_circle 
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# load dataset
#df = pd.read_csv("file1_house.csv")
df = pd.read_csv("factors.csv")

# video_file = open("house price (2).mp4", "rb")
# video_bytes = video_file.read()

# st.video(video_bytes)
video_file = open("house price (1).gif", "rb")
video_bytes = video_file.read()

st.sidebar.image(video_bytes)


class LogTransfomer(BaseEstimator, TransformerMixin):
    # def __init__(self, x,y):
    #     assert isinstance(x,y)

    #     self.x = x
    #     self.y = y
    def fit(self,x, y=None):  # always return self
        # calculate what is needed to make .transform()
        # self.mean_ = np.mean(x)
        self.n_features_in_ = x.shape[1] 
        return self # always return self
    
    def transform(self, x, y=None):
        assert self.n_features_in_ == x.shape[1]
        return np.log1p(x)

# load model
#gb_model = joblib.load("gb2_house_price.pkl")
gb_model = joblib.load("stacking_final_project.pkl")




# helper function to get gm_cluster
# def get_clust(arg1,arg2): # 2591 only not equal
        
#     if arg1 <= 1 and arg2<= 700  :
#         return 2
#     elif 1<arg1 <= 2 and 500<arg2<=1450  :
#         return 4
#     elif 2<arg1 <= 3 and 1000<arg2<=2000  :
#         return 3
#     # elif 3<arg1 <= 4 and 1000<arg2<=1600  :
#     #     return 0
#     # elif 3<arg1 <= 4 and arg2>2000  :
#     #     return 0
#     elif 3<arg1 <= 5 and arg2>1000  :
#          return 0
#     else:
#         return 3
def get_clust(arg1,arg2): # 10900 only not equal
    if arg1 == 3 and 100<arg2<=3800 :
        return 'distinctive'
    
    elif arg1 == 2 and 500<arg2<= 3600  :
        return 'economic'
    
    elif 0<=arg1<=6 and 500<arg2<= 3600  :
        return 'average'
    elif 4<=arg1<=6 and 2000<arg2<=3800 :
        return 'luxury'
    
    else:
        return 'average'


# Welcome message     
#st.title("Welcome to house price Prediction")
#st.text("information about dataset")

# inputs

columns = ['Price (in rupees)', 'location', 'Status', 'Transaction', 'Furnishing','facing', 'overlooking', 'Bathroom', 'Balcony', 'Ownership', 'BHK', 'Carpet Area Sqft', 'floor_no', 'floor_total',
       'car_parking_no', 'car_parking_type', 'house_clusters']
price_per_sqft = float(st.number_input("put price per sqft (in rupees) value: ", min_value=1.0, max_value=133929.0, format='%.2f'))
location = st.multiselect('Select location:', df['location'].unique())
Status = st.multiselect('Select Status:', df['Status'].unique())
Transaction = st.multiselect('Select Transaction:', df['Transaction'].unique())
Furnishing = st.multiselect('Select Furnishing:', df['Furnishing'].unique())
facing = st.multiselect('Select facing:', df['facing'].unique())
overlooking = st.multiselect('Select overlooking:', df['overlooking'].unique())
Bathroom = float(st.number_input("put number of Bathrooms: ", min_value=1.0, max_value=5.0, format='%.2f'))

Balcony = float(st.number_input("put number of Balcony: ", min_value=1.0, max_value=6.0, format='%.2f'))
Ownership = st.multiselect('Select Ownership:', df['Ownership'].unique())
BHK = int(st.slider("put number of bedrooms: ", min_value=1, max_value=6))
area_Sqft = float(st.number_input("put total area in sqft: ", min_value=100.0, max_value=7310.0))

floor_no = float(st.number_input("put number of Floor: ", min_value=-2.0, max_value=51.0))
floor_total = float(st.number_input("choose number of total floors: ", min_value=1.0, max_value=84.0, format='%.2f'))
car_parking_no = float(st.number_input("choose number of car parking: ", min_value=1.0, max_value=147.0, format='%.2f'))

car_parking_type = st.multiselect('Select car_parking_type:', df['car_parking_type'].unique())
#gm_clusters = get_clust((BHK,area_Sqft))
#gm_clusters = int(st.slider("choose cluster: ", min_value=0, max_value=4,help="you can press help to get suggested cluster")) # it will be object not numbers
house_clusters = st.multiselect('Select cluster:', df['house_clusters'].unique(),help="you can press help to get suggested house cluster")
cluster = st.button("help")
if cluster:
    res= get_clust(BHK,area_Sqft)
    st.write(res)






predict = st.button("Predict")
if predict:
    
    
    
    location = (location[:][0])
    Status = (Status[:][0])
    Transaction = (Transaction[:][0])
    Furnishing = (Furnishing[:][0])
    facing = (facing[:][0])
    overlooking = (overlooking[:][0])
    Ownership = (Ownership[:][0])
    car_parking_type = (car_parking_type[:][0])
    house_clusters = (house_clusters[:][0])

    #gm_clusters = np.vectorize(get_clust)(BHK,area_Sqft)
    columns = ['Price (in rupees)', 'location', 'Status', 'Transaction', 'Furnishing','facing', 'overlooking', 'Bathroom', 'Balcony', 'Ownership', 'BHK', 'Carpet Area Sqft', 'floor_no', 'floor_total',
       'car_parking_no', 'car_parking_type', 'house_clusters']
    data = pd.DataFrame([[price_per_sqft,location,Status,Transaction,Furnishing,facing, overlooking, Bathroom, Balcony, Ownership, BHK,area_Sqft,floor_no, floor_total,car_parking_no, car_parking_type, house_clusters]], columns=columns)
    #data = pd.DataFrame([[6000.0,'thane','Ready to Move','Resale','Unfurnished',	'East','Main Road' ,	1.0 ,	2.0 ,	'Freehold',1 ,500.0 ,10.0,	11.0,	1.0 ,	'Covered','average']], columns=columns)
    #data = pd.DataFrame([[10191,'chennai','Ready to Move','Resale','Semi-Furnished',	'West',"nan" ,	3.0 ,	3.0 ,	'Freehold',3 ,1400.0 ,1.0,	4.0,	1.0 ,	'Covered','distinctive']], columns=columns)
    st.dataframe(data)
    #pred = gb_model.predict(data)
    pred = np.exp(gb_model.predict(data).ravel()[0])
    st.text(f"predicted house price is : {pred} Lac")
    
    
    # columns = ['passenger_count', 'pickup_year', 'pickup_month', 'pickup_weekday', 'pickup_hour', 'pickup_season', 'pickup_period', 'distance']
    # data = pd.DataFrame([[passenger_count, pickup_year, pickup_month, pickup_weekday, pickup_hour, pickup_season, pickup_period, distance]], columns=columns)
    # st.dataframe(data)

    # # prediction
    # with st.spinner():
    #     pred = 'Please enter valid distance'
    #     if distance != 0:
    #         pred = np.exp(lr_grid.predict(data).ravel()[0])
        
    # st.text(f"Uber Fare: {pred}")
    
    
         
        