import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  RobustScaler # Scaling




# st.markdown(
#     """
#     ###  Dataset after cleaning and extract features :

#     """)
with st.expander("Dataset after cleaning and extract features"):
    df = pd.read_csv('factors.csv')
    df.drop(['Unnamed: 0'],axis=1,inplace=True)
    df_10 = pd.DataFrame(df).head(10)
    st.dataframe(df_10.style.background_gradient(cmap='Blues'))

st.subheader("uni-variate analysis for categorical features :")
uni_variate_cat = st.sidebar.multiselect("uni-variate analysis for categorical", df.select_dtypes(include='O').columns)
if not uni_variate_cat:
    df_bar = df.groupby(df['location'])[['Bathroom']].count().reset_index().sort_values(by='Bathroom', ascending=False).head(20)
    fig4=px.bar(x=df_bar['location'], y=df_bar['Bathroom'],color_discrete_sequence=['#de425b','#ef805b','#f7b672'],title="location")
    st.write(fig4)
else:
    col = uni_variate_cat[:][0]
    if df[col].nunique() <= 7:
        fig_han = plt.figure(figsize=(10,5)) 
        labels = df[col].value_counts().index
        values = df[col].value_counts().values
        fig_han = px.pie(data_frame=df,values=values, names = labels,labels=values)
        fig_han.update_traces(textposition='inside', textinfo='percent+label')
        fig_han.update_layout(
            title_text = col,
            legend_title = col,
            )
        st.write(fig_han)
    elif df[col].nunique() > 7 and df[col].nunique() < 35:
        fig4=px.bar(x=df[col],color_discrete_sequence=['#de425b','#ef805b','#f7b672'],title=uni_variate_cat[:][0])
        st.write(fig4)
    else:
        df_bar = df.groupby(df[col])[['Bathroom']].count().reset_index().sort_values(by='Bathroom', ascending=False).head(20)
        fig4=px.bar(x=df_bar[col], y=df_bar['Bathroom'],color_discrete_sequence=['#de425b','#ef805b','#f7b672'],title=col)
        st.write(fig4)
st.subheader("uni-variate analysis for numeric features :")
uni_variate_num = st.sidebar.multiselect("uni-variate analysis for numeric", df.select_dtypes(include='number').columns)
if not  uni_variate_num:
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # sns.histplot(df['Price (in rupees)'], kde=True, ax=axes[0])
    # sns.boxplot(df['Price (in rupees)'], ax=axes[1])
    # st.pyplot(plt)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    sns.histplot(df['Price (in rupees)'],bins=50, kde=True, ax=ax1,color="brown")
    sns.boxplot(df['Price (in rupees)'], ax=ax2,color="green")
    st.pyplot(plt)
else:
    col_num = uni_variate_num[:][0]
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    sns.histplot(df[col_num], kde=True,bins=50 ,ax=ax1,color="brown")
    sns.boxplot(df[col_num], ax=ax2,color="green")
    st.pyplot(plt)
st.subheader("bi-variate analysis  :")
questions = ['relation between price and area','relation between price and facing','mean price for top 20 locations','mean price for each house cluster']

graph_choice = st.sidebar.multiselect("bi-variate analysis", questions)
if not graph_choice:
    fig = plt.figure(figsize=(12, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True)
    st.pyplot(plt)

else:
    col_graph = graph_choice[:][0]
    def func1():
            
            # fig = plt.figure(figsize = (8, 4))
            # #palette =sns.color_palette("hls", 8)
            # #palette = sns.color_palette("Paired")#("rocket")#("mako_r", 6)
            # palette = sns.color_palette("Set2")
            # plt.title("relation between price and area")
            # sns.lineplot(x=df['Carpet Area Sqft'], y=df['Amount (Lac)'],palette=palette,ci=None,hue=df['Furnishing'])#ci for shadow # palette works with hue only
            # st.pyplot(plt)
            data1 = px.scatter(df, x = "Carpet Area Sqft", y = "Amount (Lac)",size = "BHK" ,color="house_clusters")
            data1['layout'].update(title="Relationship between Area and Price using Scatter Plot.",
                    titlefont = dict(size=20),xaxis = dict(title="Area",titlefont=dict(size=19)),
                    yaxis = dict(title = "price", titlefont = dict(size=19)))
            st.plotly_chart(data1)
    def func2():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
            #sns.violinplot(x=df['facing'],y=df['Amount (Lac)'] ,ax=ax1,color="yellow")
            sns.boxplot(x=df['facing'],y=df['Amount (Lac)'] ,ax=ax1,color="green")
            sns.stripplot(x=df['facing'],y=df['Amount (Lac)'],ax=ax2,color = "red")
            plt.xticks(rotation=45, ha='right')
            plt.title("Relation between price and facing")
            st.pyplot(plt)
        
    def func3():
            graph = df.groupby("location")[['Amount (Lac)']].mean().sort_values(by='Amount (Lac)', ascending=False).rename(columns={'Amount (Lac)':'price'}).reset_index().head(20)
            fig = plt.figure(figsize=(12,6))
            plt.bar(graph['location'], graph['price'], color ='maroon', width = 0.4)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel("location")
            plt.ylabel("mean price (Lac)")
            plt.title("the mean price for 20 top location")
            st.pyplot(plt)
            
    def func4():
            graph = df.groupby("house_clusters")[['Amount (Lac)']].mean().sort_values(by='Amount (Lac)', ascending=False).rename(columns={'Amount (Lac)':'price'}).reset_index()
            #fig = plt.figure(figsize=(12,6))
            # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
            # sns.boxplot(x=graph['house_cluster'],y=graph['price'] ,ax=ax1,color="green")
            # sns.stripplot(x=graph['house_cluster'],y=graph['price'],ax=ax2,color = "red")
            # plt.xticks(rotation=45, ha='right')
            # plt.title("mean price for each house cluster")
            # st.pyplot(plt)
            fig = plt.figure(figsize=(12,6))
            plt.bar(graph['house_clusters'], graph['price'], color ='#ef805b', width = 0.4)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel("house cluster")
            plt.ylabel("mean price (Lac)")
            plt.title("the mean price for each house cluster")
            st.pyplot(plt)

    draw_graph = {'relation between price and area':func1, "relation between price and facing":func2,"mean price for top 20 locations":func3,"mean price for each house cluster":func4}
    draw_graph[col_graph]()
st.subheader("multi-variate analysis:")
multi_options=["multi-variate analysis"]
multi_variate = st.sidebar.multiselect("multi-variate analysis",multi_options)
if not multi_variate:
     st.write("👈 choose multi_variate")
else:
    plot = sns.pairplot(df.select_dtypes(include='number'))    
    st.pyplot(plot)

st.subheader("numeric features after preprocessing:")
class LogTransfomer(BaseEstimator, TransformerMixin):
    
    def fit(self,x, y=None):  # always return self
        # calculate what is needed to make .transform()
        # self.mean_ = np.mean(x)
        self.n_features_in_ = x.shape[1] 
        return self # always return self
    
    def transform(self, x, y=None):
        assert self.n_features_in_ == x.shape[1]
        return np.log1p(x)
log_transformer = LogTransfomer()
cols = ['Price (in rupees)','Carpet Area Sqft']
after_pre = st.sidebar.multiselect("numeric features after preprocessing",cols)

if not  after_pre:
    st.write("👈 choose numeric features after preprocessing")
else:
    
    col_item = after_pre[:][0]
       
    data_imputed = SimpleImputer(strategy='median').fit_transform(df[[col_item]])
    data_logged = log_transformer.fit_transform(data_imputed)
    rbs_scaler = RobustScaler()
    data_scaled = rbs_scaler.fit_transform(data_logged)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    sns.histplot(data_scaled,bins=50, kde=True, ax=ax1,color="brown")
    sns.boxplot(data_scaled, ax=ax2,color="green")
    st.pyplot(plt)
    
        