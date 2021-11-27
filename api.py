# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:10:13 2021

@author: Jeff
"""

from flask
import Flaskapp = Flask(__name__)

if __name__ == '__main__':
     app.run(port=8080)

# import requests

# url = 'http://localhost:5000/results'
# r = requests.post(url,json={'rate':5, 'sales_in_first_month':200, 'sales_in_second_month':400})

# print(r.json())