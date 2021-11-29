# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:10:13 2021

@author: Jeff
"""

from flask import Flask, jsonify
import requests
import sys
import joblib

app = Flask(__name__)

MODEL_URL = "https://github.com/jtahiata/P7_tahiata_jeff_A/blob/main/loan_model.joblib?raw=true"

@app.route('/api/model/')
def get_model():
    r = requests.get(MODEL_URL)
    model = joblib.load(r)

    if r.status_code != 200:
        return jsonify({
            'status': 'error',
            'message': 'La requête à l\'API météo n\'a pas fonctionné. Voici le message renvoyé par l\'API : {}'.format(model)
        }), 500
    
    return model

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 8000 # If you don't provide any port the port will be set to 12345

    app.run(port=port, debug=True)

