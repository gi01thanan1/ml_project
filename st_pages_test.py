import streamlit as st
#from PIL import Image
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
import joblib
#from Prediction import LogTransfomer

# dark = '''
# <style>
#     .stApp {
#     background-color: black;
#     }
# </style>
# '''
# st.markdown(dark, unsafe_allow_html=True)
hide_streamlit_style = """

    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
    
    
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.image("house_3.jpg")


# bottom_image = st.file_uploader('house_3', type='jfif', key=6)
# if bottom_image is not None:

class LogTransfomer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):  # always return self
        # calculate what is needed to make .transform()
        # self.mean_ = np.mean(x)
        self.n_features_in_ = x.shape[1] 
        return self # always return self
    
    def transform(self, x, y=None):
        assert self.n_features_in_ == x.shape[1]
        return np.log1p(x)
# image = Image.open('house_3.jfif')
# new_image = image.resize((700, 200))
# st.image(new_image)
# st.write(Image.__version__)
pg = st.navigation([st.Page("Home.py",title="Welcome", icon="👋"), st.Page("EDA and Visualization.py", icon="📈"),st.Page("Prediction.py",icon="🏡")])#":material/target:")])
#pages/1_📈_Plotting_Demo.py
pg.run()

#df = pd.read_csv("file1_house.csv")


# if __name__ == '__main__':
#     LogTransfomer()
#     class LogTransfomer(BaseEstimator, TransformerMixin):

#         def fit(self, x, y=None):  # always return self
#             # calculate what is needed to make .transform()
#             # self.mean_ = np.mean(x)
#             self.n_features_in_ = x.shape[1] 
#             return self # always return self
        
#         def transform(self, x, y=None):
#             assert self.n_features_in_ == x.shape[1]
#             return np.log1p(x)
#     # load model
#     gb_model = joblib.load("gb2_house_price.pkl")