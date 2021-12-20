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
import joblib
from io import BytesIO
        
# Shap init JavaScript visualization code to notebook
shap.initjs()

# 1) Import initialisation

st.title('Loan Prediction')
test = 'test.csv'
test_original = 'application_test.csv'
df_feat = 'HomeCredit_columns_description.csv'

# Standardize database
df = pd.read_csv(test)
df_columns = df.columns[1:]
df_columns_bool = df.iloc[:,1:].loc[:,df.nunique() == 2].columns
df_columns_nbool = list(set(df_columns) - set(df_columns_bool))
df_columns_nbool.sort()

# Original database
df_original = pd.read_csv(test_original)


df_features = pd.read_csv(df_feat, low_memory=False, encoding='latin-1')

option = st.sidebar.selectbox("Which application ?",
                              ('Display database','Solvability prediction',
                                'Crossed features', 'Corr features'))

# 2) Functions

def summary():
    
    mLink = 'https://github.com/jtahiata/P7_tahiata_jeff_A/blob/main/loan_model.joblib?raw=true'
    mfile = BytesIO(requests.get(mLink).content)
    model = joblib.load(mfile)
    exp = shap.TreeExplainer(model)
    shap_val = exp.shap_values(df.iloc[:,1:].values)[0]
    
    st.subheader('Summary Plot')
    fig, ax = plt.subplots()
    shap.summary_plot(shap_val, df.iloc[:,1:])
    st.pyplot(fig)
    st.write('The summary plot combines feature importance with feature effects. Each point on the summary plot is a Shapley value for a feature and an instance. The position on the y-axis is determined by the feature and on the x-axis by the Shapley value. The color represents the value of the feature from low to high. Overlapping points are jittered in y-axis direction, so we get a sense of the distribution of the Shapley values per feature. The features are ordered according to their importance.')

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
    
    st.subheader('Database')
    st.write(len(df),'customers inside the database')
    summary_btn = st.sidebar.button('Display Summary plot')
    
    if summary_btn:
        summary()
        
    st.write('Original database : 100 first customers')
    st.write(df_original.head(100))
    
    st.write('Standardized database : 100 first customers')
    st.dataframe(df.iloc[:,1:].head(100))
    
    with st.expander("More infomation about features:"):
        st.table(df_features.iloc[:,1:].sort_values('Row'))
    
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
                                ('Force plot',
                                  'Decision plot'))
    predict_btn = st.sidebar.button('Predict acceptability')
    st.subheader('Solvability prediction')

    acceptability = predict['Prediction'][0]
    probability = predict['Probability'][0][0]      
        
    # Calculate Shap values
    expected_value = predict['Expected_value']
    shap_values = np.fromiter(json.loads(predict['Shap_values'])["0"].values(), dtype=float)
    
    if predict_btn:
        
        if acceptability == 0:
            st.write('Customer will refund the loan on time with a score of ',
                         round(probability, 3))
        elif acceptability == 1:
            st.write('Customer will not refund the loan on time with a probability of ',
                         round(1 - probability, 3))
            
        if plot == 'Force plot':
            force()
                
        if plot =='Decision plot':
            decision()

        with st.expander("More infomation about features:"):
            st.table(df_features.iloc[:,1:].sort_values('Row'))

# 5) General statistics

if option == 'Crossed features':
        
    feat1 = st.sidebar.selectbox("1st feature (none bool)?",
                                df_columns_nbool)
    
    feat2 = st.sidebar.selectbox("2nd feature (none bool)?",
                                df_columns_nbool)
    
    feat3 = st.sidebar.selectbox("3rd feature (bool)?",
                                (df_columns_bool.sort_values()))
    
    cross_btn = st.sidebar.button('Crossed features')
    
    if cross_btn:
        
        st.subheader('Crossed features')
        fig = px.scatter(x = df.loc[:,feat1], y = df.loc[:,feat2],
                         color = df.loc[:,feat3], 
                         labels={"x": str(feat1), "y": str(feat2), "color": str(feat3)})
        plt.xlabel(str(feat1))
        plt.ylabel(str(feat2))
        st.plotly_chart(fig)

        with st.expander("More infomation about features:"):
            st.table(df_features.iloc[:,1:].sort_values('Row'))
    
if option == 'Corr features':      
    
    st.subheader('Correlation matrix on non bool features')
    fig = px.imshow(df.loc[:,df_columns_nbool].corr())
    st.plotly_chart(fig)

    with st.expander("More infomation about features:"):
        st.table(df_features.iloc[:,1:].sort_values('Row'))