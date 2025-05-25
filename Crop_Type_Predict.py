import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("crop.csv")
X = data.drop("label",axis=1)
y = data["label"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_tr , x_val , y_tr , y_val = train_test_split(X_train,y_train,test_size=0.1,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
x_tr = scaler.fit_transform(x_tr)
x_val = scaler.transform(x_val)
x_test = scaler.transform(X_test)

st.title("Crop Prediction")
st.subheader("Enter the values")
st.subheader("N")
st.subheader("P")
st.subheader("K")
st.subheader("temperature")
st.subheader("humidity")
st.subheader("ph")
st.subheader("rainfall")
model = joblib.load('crop.pkl')
N = st.number_input("N")
P = st.number_input("P")
K = st.number_input("K")
temperature = st.number_input("temperature")
humidity = st.number_input("humidity")
ph = st.number_input("ph")
rainfall = st.number_input("rainfall")

if st.button("Predict"):
    result = model.predict([[N,P,K,temperature,humidity,ph,rainfall]])
    st.success(result)
    st.balloons()
    st.write("The predicted crop is",result)
    st.write("You can check the accuracy of the model below")

    st.write("The accuracy of the model is",model.score(x_val,y_val))