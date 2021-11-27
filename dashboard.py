# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:45:47 2021

@author: Jeff
"""
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
            
# Shap init JavaScript visualization code to notebook
shap.initjs()
# from flask import Flask
# app = Flask(__name__)


# 1) Import initialisation

st.title('Loan Prediction')
test = 'test.csv'
test_original = 'application_test.csv'
df_feat = 'HomeCredit_columns_description.csv'
model = joblib.load('loan_model.joblib')

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

def customer_selection():

    st.sidebar.subheader('Customer selection')
    customer = st.sidebar.selectbox('Customer ID',df_original['SK_ID_CURR'])
    idx = df[df_original['SK_ID_CURR'] == int(customer)].index
    customer_data = df.iloc[idx,1:] # 1er colonne = index & iloc car import
    return customer_data

def summuary():
    
    st.subheader('Figure 1 : Summary plot')
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, feature_names = df_columns)
    st.pyplot(fig)
    st.write('This diagram represents the distribution of shap values for each entity in the data set.')

def force():
    
    st.subheader('Figure 2 : Force plot')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.force_plot(expected_value, shap_values[1],
                    feature_names = df_columns, link='logit',
                    matplotlib=True, figsize=(12,3))
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()
    st.write('This graph shows the path the model took for a particular decision based on the shap values of individual features. The individual plotted line represents a sample of data and how it reached a particular prediction.')
    st.write('There are several use cases for a decision plot. We present several cases here. 1. Show a large number of feature effects clearly. 2. Visualize multioutput predictions. 3. Display the cumulative effect of interactions. 4. Explore feature effects for a range of feature values. 5. Identify outliers. 6. Identify typical prediction paths. 7. Compare and contrast predictions for several models.')

def decision():
    
    st.subheader('Figure 3: Decision Plot')
    fig, ax = plt.subplots()
    shap.decision_plot(expected_value, shap_values[1], df_columns,
                                  link='logit', highlight=0)
    st.pyplot(fig)
    st.write('It plots the shap values using an additive strength layout. Here we can see which features contributed most positively or negatively to the prediction.')


# 3) Shap : Create object that can calculate shap values based on a tree model

explainer = shap.TreeExplainer(model)
expected_value = explainer.expected_value[1]

# if isinstance(expected_value, np.ndarray):
#     expected_value = expected_value[1]
    
# 4) Display database

if option == 'Display database':
        
    st.write('Shap expected value:',expected_value)
    
    nb = st.sidebar.number_input('Datafile lines to display', min_value=1,
                                 value=1, step=1)
    st.write('Original database')
    st.dataframe(df_original.head(int(nb)))
    st.write('Standard database')
    st.dataframe(df.head(int(nb)))
    
# 5) Solvability prediction

if option == 'Solvability prediction':
    
    st.subheader('Customer selected')
    st.write(customer_data)
    customer_data = customer_selection()
    plot = st.sidebar.selectbox("Which plot ?",
                                ('Summary plot','Force plot',
                                 'Decision Plot'))
    acceptability = model.predict(customer_data)
    probablity = float(model.predict_proba(customer_data)[:,1])
    predict_btn = st.sidebar.button('Predict acceptability')
    st.subheader('Solvability prediction')
    
    # Calculate Shap values
    shap_values = explainer.shap_values(customer_data)        
    
    if predict_btn:
        if acceptability == 0:
            st.write('Customer will refund the loan on time with a probability of ',
                     round(100 - probablity*100, 1),"%")
        elif acceptability == 1:
            st.write('Customer will not refund the loan on time with a probability of ',
                     round(probablity*100, 1),"%")
            
        st.write('Shap values',shap_values[1])
        
    if plot == 'Summary plot':
        summuary()
            
    if plot == 'Force plot':
        force()
            
    if plot =='Decision Plot':
        decision()
        
# 6) General statistics

if option == 'General statistics':
        
    feat1 = st.sidebar.selectbox("1st feature ?",
                                (df_columns))
    
    feat2 = st.sidebar.selectbox("2nd feature ?",
                                (df_columns))
    
    stat_btn = st.sidebar.button('Customer statistics')
    
    if stat_btn:
        
        st.subheader('Crossed stats between features')
        fig, ax = plt.subplots()
        ax.scatter(x = df_original.loc[:,feat1], y = df_original.loc[:,feat2])
        plt.xlabel(str(feat1))
        plt.ylabel(str(feat2))
        st.pyplot(fig)
    
with st.expander("More infomation about features:"):
    st.table(df_features)

# import requests

# url = 'http://localhost:5000/results'
# r = requests.post(url,json={'rate':5, 'sales_in_first_month':200, 'sales_in_second_month':400})

# print(r.json())