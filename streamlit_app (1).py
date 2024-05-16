#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import streamlit as st
import pandas as pd
import sklearn

st.title(':blue[Heart disease Prediction App]')
st.write("""-- This app predicts a Patient has a heart disease or not --

""")
st.write(':point_left: (click arrow sign for hide and unhide form) :green[Please Fillup the input field of left side for Prediction.] :sunglasses:')
st.download_button('Download Sample file link for check', 'https://github.com/ripon2488/heart-disease-prediction-machine-learning/blob/main/heart_disease_dataset.csv')
st.sidebar.header('Please Input Features Value')


def user_input_features():
    age = st.sidebar.number_input('Age of persons: ')
    #sex = st.sidebar.selectbox('Gender of persons 0=Female, 1=Male: ',(0,1))
    bp = st.sidebar.number_input('Resting Blood Pressure: ')
    chol = st.sidebar.number_input('Serum cholestrol in mg/dl: ')
    maxhr = st.sidebar.number_input('Maximum heart rate achieved: ')
    oldpeak = st.sidebar.number_input(' ST depression induced by exercise relative to rest (oldpeak): ')
    

    data = {'age':age, 'bp':bp, 'chol':chol, 'maxhr':maxhr,
        'oldpeak':oldpeak,}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

st.write(input_df)
def predict(data):
    clf = pickle.load(open("final_model.pkl",'rb'))
    return clf.predict(data)

if st.button("Click here to Predict type of Disease"):
    result = predict(input_df)
    if (result== 0):
        st.subheader('The Person :green[does not have a Heart Disease] :sunglasses: 	:sparkling_heart:')
    else:
        st.subheader('The Person :red[has Heart Disease] :worried:')

