# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import joblib

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index_logistic.html')
    if request.method == 'POST':
        
        # 파라미터를 전달 받습니다.
        weight = float(request.form.get('weight',False))
        length = float(request.form.get('length',False))
        diagonal = float(request.form.get('diagonal',False))
        height = float(request.form.get('height',False))
        width = float(request.form.get('width',False))

        # 모델 가져오기
        #model = joblib.load("./model/logistic_model.pkl") # 로지스틱 회귀모델
        model = joblib.load("./model/SGD_model.pkl") # 확률적 경사하강법 모델
           
        # 제품 무게를 예측합니다.
        result = 0
        result = model.predict([[weight, length, diagonal, height, width]])

        return render_template('index_logistic.html', result = result)

if __name__ == '__main__':
   app.run(debug = True)