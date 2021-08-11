# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 16:46:50 2021

@author: Mehrnoush
"""

from flask import Flask, request
import pickle

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Pinging Model Application!!!"

@app.route('/show', methods=["GET"])
def show():
    user_id = 'A'
    item_id = '1'
    loaded_svd_algo = pickle.load(open('svd_model.pkl', 'rb'))
    prediction = loaded_svd_algo.predict(uid=user_id, iid=item_id)
    return str(prediction)

@app.route('/sum', methods=["POST"])
def hello_sum():
    value1=int(request.form['num1'])
    value2=int(request.form['num2'])
    return (str(value1+value2))

@app.route('/predict', methods=["POST"])
def predict():

    user_id = 'A'
    item_id = request.form['num2']
    loaded_svd_algo = pickle.load(open('svd_model.pkl', 'rb'))
    prediction = loaded_svd_algo.predict(uid=user_id, iid=item_id)

    return (str(prediction))

if __name__ == "__main__":
    app.run(debug=True)