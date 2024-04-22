#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np 
import pandas as pd
import joblib
import pickle
import streamlit as st
from sklearn.preprocessing import *

st.title('Heart disease Prediction App')
st.write("""-- This app predicts a Patient has a heart disease or not --

""")
#st.download_button('Download Sample file link for check', 'https://github.com/ripon2488/heart-disease-prediction-machine-learning/blob/main/heart_disease_dataset.csv')
st.sidebar.header('Please Input Features Value')

# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Age of persons: ')
    bp = st.sidebar.number_input('Resting Blood Pressure: ')
    chol = st.sidebar.number_input('Serum cholestrol in mg/dl: ')
    maxhr = st.sidebar.number_input('Maximum heart rate achieved: ')
    oldpeak = st.sidebar.number_input(' ST depression induced by exercise relative to rest (oldpeak): ')
    

    data = {'Age':age,'RestingBP':bp, 'Cholesterol':chol, 'MaxHR':maxhr,
        'Oldpeak':oldpeak,}
    #Age	RestingBP	Cholesterol	MaxHR	Oldpeak
    features = pd.DataFrame(data, index=[0])
    #features=StandardScaler().fit_transform(features)
    return features
input_df = user_input_features()

st.write(input_df)

def predict(data):
    clf = pickle.load(open(r"final_model_LogR (1).pkl", 'rb'))
    #joblib.load("model_LogR.sav")
    return clf.predict(data)

if st.button("Click here to Predict type of Disease"):
    result = predict(input_df)
    st.write(result)
    if (result[0]== 0):
        st.success('The Person does not have a Heart Disease :sunglasses: 	:sparkling_heart:')
        st.balloons()
    else:
        st.error('The Person has Heart Disease :worried:')
st.subheader('Developed by Dharani J S')

