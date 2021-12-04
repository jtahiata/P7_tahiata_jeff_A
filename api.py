# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:10:13 2021

@author: Jeff
"""

from flask import Flask, request, jsonify
import joblib
import shap

app = Flask(__name__)

MODEL = joblib.load("https://github.com/jtahiata/P7_tahiata_jeff_A/blob/main/loan_model.joblib?raw=true")
CUSTOMER_FEAT = "https://github.com/jtahiata/P7_tahiata_jeff_A/blob/main/customer_features.csv?raw=true"

@app.route('/api/predict/')
def predict():
    features = []
    for i in range(CUSTOMER_FEAT):
        features.append(request.args.get(CUSTOMER_FEAT[i]))
    customer_class = MODEL.predict(features)
    # Create and send a response to the API caller
    return jsonify(status='complete', customer_class=customer_class)

@app.route('/api/score/')
def score():
    features = []
    for i in range(CUSTOMER_FEAT):
        features.append(request.args.get(CUSTOMER_FEAT[i]))
    customer_score = MODEL.predict_proba(features)[:,1]
    # Create and send a response to the API caller
    return jsonify(status='complete', customer_score=customer_score)

@app.route('/api/explainer/')
def explainer():
    exp = shap.TreeExplainer(MODEL)
    # Create and send a response to the API caller
    return jsonify(status='complete', explainer=exp)

if __name__ == '__main__':
    port = 8000
    app.run(port=port, debug=True)
