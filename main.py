# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import joblib

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        
        # 파라미터를 전달 받습니다.
        width = float(request.form.get('width',False))        

        # 모델 가져오기
        model = joblib.load("C:/Users/MAYA/PycharmProjects/Machin-Learning/model/polynomial_model.pkl")
           
        # 제품 무게를 예측합니다.        
        weight = 0
        weight = model.predict([[width**2,width]])

        return render_template('index.html', weight=weight)

if __name__ == '__main__':
   app.run(debug = True)