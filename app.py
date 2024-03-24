import numpy as np
import scipy as sp
import pandas as pd 
import os
import matplotlib.pyplot as plt
from flask import Flask,request,render_template
import pickle


app = Flask(__name__)
model = pickle.load (open ('model.pkl','rb'))

@app.route ('/')
def home():
    return render_template ('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    file = request.files['file']
    file.save(os.path.join('C:/Users/SK2712/OneDrive - Cal State Fullerton/Desktop/FinalProj/finalProject', file.filename))
    df = pd.read_csv(file.filename)

    # df['Attrition'].replace('Yes','1',inplace=True)
    # df['Attrition'].replace('No','0',inplace=True) 
    df.drop(['EmployeeCount','Over18','StandardHours','EmployeeNumber'],axis=1,inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes

    # X = df.drop(['Attrition'], axis=1)
    X = df

    prediction = model.predict(X)

    print(prediction)

    count= np.count_nonzero(prediction == 1)
    co = len(prediction)
    for i in range(0, len(prediction)):  
        print(prediction[i]),
    print(count)
    print(co)
    p = count / co * 100

    if prediction.any():
        return render_template('show.html',p=p,prediction_text='Predicted Sucessfully')
    else:
        return render_template ('index.html',prediction_text='Predicted Failed')

           