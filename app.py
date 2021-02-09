# Import the Libraries
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps # Image processing
import pickle
import sklearn
from sklearn.model_selection import train_test_split
import base58
#####################################################################################################################
def download_link(object_to_download, download_filename, download_link_text):
   
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base58.b58encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
#####################################################################################################################

#Create a title and a sub-title
st.write("""
# Universal ML Regressor
Sci Kit Learn Based ML Web App for Regression Applications
""")

#Open and Display an Image
#image = Image.open('/content/gdrive/My Drive/Machine Learning Web Application/Diabetes Detection/Diabetes Detection.png')
#st.image(image, use_column_width=True) # caption = 'ML', 

st.write("import csv dataset, that does not have any string predictors or missing values")
st.write("csv file must have target variable in the rightmost column, and the remaining predictors to the left")

# COMPUTATION
st.sidebar.header('User Input Dataset and Parameters')


st.sidebar.markdown("""
[Example CSV input file](https://drive.google.com/file/d/16iVua1vtUVvRno8lmTH_ZQ7f964OXDiU/view?usp=sharing)
""")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

split_ratio = st.sidebar.slider('Train Test Split Ratio (Input Test Percentage)', 0, 100, 80, 10)
split_ratio = split_ratio/100
model = st.selectbox('ML Regressor Model', ('Linear Regression', 'Polynomial Regression', 'SVM Support Vector Machine', 'Decision Tree', 'Random Forest', 'XG Boost'))


if uploaded_file is not None:

  dataset = pd.read_csv(uploaded_file)
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, -1].values

  #Splitting the dataset into Training and Testing Set
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state = 0)

  if model == 'Linear Regression':
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
  elif model == 'Polynomial Regression':
    poly_degree = st.sidebar.slider('Polynomial Regressor Degree', 0, 5, 2)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X, y)
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = poly_degree)
    X_poly = poly_reg.fit_transform(X)
    regressor = LinearRegression()
    regressor.fit(X_poly, y)
  elif model == 'SVM Support Vector Machine':
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X, y)
  elif model == 'Decision Tree':
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X, y)
  elif model == 'Random Forest':
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X, y)
  elif model == 'XG Boost':
    from xgboost import XGBRegressor
    regressor = XGBRegressor()
    regressor.fit(X_train, y_train)
  else:
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

  st.subheader('Trained Model Results')
  #from sklearn.metrics import accuracy_score
  #y_pred = regressor.predict(X_test)
  #accuracy = accuracy_score(y_test, y_pred)
  #st.write('Accuracy:')
  #st.success(str(accuracy))

  st.subheader('Downladable Model')
  st.write('Dowload regressor model, and use the following code to create your own ML products')
  pickle.dump(regressor, open('regressor.pkl', 'wb'))
  
  if st.button('Get Link to Download Trained Model as pkl'):
    tmp_download_link = download_link('regressor.pkl', 'regressor.pkl', 'Click here to download your Trained Model!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
  
  st.subheader('Code')
  if st.button('Get Link to Python (.py) file to make ML predictions with your model!'):
    tmp_download_link = download_link('/content/gdrive/MyDrive/Machine Learning/Universal Regressor/Predictor-Universal_Regressor.ipynb', 'app.py', 'Click here to download your Python Code!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

  st.subheader('Requirements')
  if st.button('Get Link to requirements.txt file and Procfile to deploy your ML Model as a website'):
    tmp_download_link = download_link('/content/gdrive/MyDrive/Machine Learning/Universal Regressor/requirements.txt', 'requirements.txt', 'Click here to download your requirements.txt!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
    tmp_download_link = download_link('/content/gdrive/MyDrive/Machine Learning/Universal Regressor/Procfile', 'Procfile', 'Click here to download the Procfile!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

  st.success('Create a file named "input.csv" and place it in the same folder as the regressor model and the Code to make Predictions')
