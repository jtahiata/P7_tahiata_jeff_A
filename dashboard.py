# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:45:47 2021

@author: Jeff
"""

import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
# import plotly.graph_objects as go
import requests
import json
        
# Shap init JavaScript visualization code to notebook
shap.initjs()

# 1) Import initialisation

st.title('Loan Prediction')
test = 'test.csv'
test_original = 'application_test.csv'
df_feat = 'HomeCredit_columns_description.csv'
# model = joblib.load('loan_model.joblib')

df = pd.read_csv(test)

df_original = pd.read_csv(test_original)
df_columns = df.columns[1:]
df_features = pd.read_csv(df_feat, low_memory=False, encoding='latin-1')

option = st.sidebar.selectbox("Which application ?",
                              ('Display database','Solvability prediction',
                                'General statistics'))

st.subheader('Database')
st.write(len(df),'customers inside the database')

customer_data = df.iloc[0,1:]

# 2) Functions

def customer_idx():

    st.sidebar.subheader('Customer selection')
    customer = st.sidebar.selectbox('Customer ID',df_original['SK_ID_CURR'])
    idx = df[df_original['SK_ID_CURR'] == int(customer)].index
    return idx

def force():
    
    st.subheader('Figure 1 : Force plot')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.force_plot(expected_value, shap_values,
                    feature_names = df_columns, link='logit',
                    matplotlib=True, figsize=(12,3))
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()
    st.write('This graph shows the path the model took for a particular decision based on the shap values of individual features. The individual plotted line represents a sample of data and how it reached a particular prediction.')
    st.write('There are several use cases for a decision plot. We present several cases here. 1. Show a large number of feature effects clearly. 2. Visualize multioutput predictions. 3. Display the cumulative effect of interactions. 4. Explore feature effects for a range of feature values. 5. Identify outliers. 6. Identify typical prediction paths. 7. Compare and contrast predictions for several models.')

def decision():
    
    st.subheader('Figure 2: Decision Plot')
    fig, ax = plt.subplots()
    shap.decision_plot(expected_value, shap_values, df_columns,
                                  link='logit', highlight=0)
    st.pyplot(fig)
    st.write('It plots the shap values using an additive strength layout. Here we can see which features contributed most positively or negatively to the prediction.')
    
# 3) Display database

if option == 'Display database':
        
    st.write('Original database')
    st.write(df_original.head(100))
    
    st.write('Standard database')
    st.dataframe(df.head(100))
    
    with st.expander("More infomation about features:"):
        st.table(df_features.iloc[:,1:])
    
# 4) Solvability prediction

if option == 'Solvability prediction':
    
    st.subheader('Customer selected')
    idx = customer_idx()
    
    id_curr={'SK ID CURR':int(idx.values)}
    
    customer_data = df.iloc[idx,1:] # 1er colonne = index & iloc car import
    json_customer = json.loads(customer_data.to_json(orient='records'))[0]
    data_json = {'data': json_customer}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    
    url = 'http://jtahiata.pythonanywhere.com/predict'
    r = requests.post(url, json=data_json, headers=headers)
    st.write(data_json)
    predict = json.loads(r.content.decode("utf-8"))
    
    plot = st.sidebar.selectbox("Which plot ?",
                                ('Summary plot','Force plot',
                                  'Decision Plot'))
    predict_btn = st.sidebar.button('Predict acceptability')
    st.subheader('Solvability prediction')

    acceptability = predict['Prediction'][0]
    probability = predict['Probability'][0][0]      
        
    # Calculate Shap values
    expected_value = predict['Expected_value']
    shap_values_ = predict['Shap_values']
    shap_values = np.array(shap_values_)
    
    st.write(acceptability)
    st.write(probability)
    st.write(expected_value)
    st.write(shap_values)
    
    if predict_btn:
        
        if acceptability == 0:
            st.write('Customer will refund the loan on time with a probability of ',
                         round(probability*100, 1),"%")
        elif acceptability == 1:
            st.write('Customer will not refund the loan on time with a probability of ',
                         round((1 - probability)*100, 1),"%")
            
        if plot == 'Force plot':
            force()
                
        if plot =='Decision Plot':
            decision()

# 5) General statistics

if option == 'General statistics':
        
    feat1 = st.sidebar.selectbox("1st feature ?",
                                (df_columns))
    
    feat2 = st.sidebar.selectbox("2nd feature ?",
                                (df_columns))
    
    stat_btn = st.sidebar.button('Customers statistics')
    
    if stat_btn:
        
        st.subheader('Crossed stats between features')
        fig = px.scatter(x = df_original.loc[:,feat1], y = df_original.loc[:,feat2])
        plt.xlabel(str(feat1))
        plt.ylabel(str(feat2))
        st.plotly_chart(fig)