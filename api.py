# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:10:13 2021

@author: Jeff
"""

from flask import Flask, request, jsonify
import joblib
import shap
import pandas as pd

app = Flask(__name__)

model = joblib.load('/home/jtahiata/mysite/functions/loan_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data_json = data['data']
    data_val = [list(data_json.values())]
    df = pd.json_normalize(data_json)

    # print(data_val)
    predict_val = model.predict(data_val)
    prediction = predict_val.tolist()
    # print(prediction)
    proba_val = model.predict_proba(data_val)
    probability = proba_val.tolist()
    # print(probability)

    # Shap init JavaScript visualization code to notebook, Calculate Shap values
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    expected_value = explainer.expected_value[1]
    shap_values_ = explainer.shap_values(df.values)[0][0]
    shap_values = []

    for string in shap_values_:
        new_string = float(str(string).replace("#12", ""))
        shap_values.append(new_string)

    print(data_json)
    print(df.values)
    print(shap_values)

    return jsonify(Prediction=prediction, Probability=probability, Expected_value=expected_value, Shap_values=shap_values)

if __name__ == '__main__':
    app.run(debug=True)
