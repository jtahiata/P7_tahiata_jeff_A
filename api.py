# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:10:13 2021

@author: Jeff
"""

from flask import Flask, request, jsonify, Response
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('/home/jtahiata/mysite/functions/loan_model.joblib')

@app.route('/predict', methods=['GET','POST'])
def predict():
    data = request.get_json()
    data_json = data['data']
    data_val = [list(data_json.values())]
    predict_val = model.predict(data_val)
    prediction = predict_val.tolist()
    proba_val = model.predict_proba(data_val)
    probabilite = proba_val.tolist()

    return jsonify(Prediction=prediction, Probabilite=probabilite)
    # print(data)
    # return Response(status=200)
    # prediction = np.array2string(model.predict(data))
    # data = json.loads(elevations)

    # prediction = model.predict(pd.DataFrame.from_dict(data, orient="index").values)

    # return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)