import boto3
import pandas as pd 

from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
s3 = boto3.client('s3')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

def get_file_url(bucket_name, file_key):
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': bucket_name,
            'Key': file_key
        },
        ExpiresIn=3600  
    )
    return url

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return 'No file provided in the request'

    file.save(file.filename)
    bucket_name = 'employeeattri1'  
    s3.upload_file(file.filename, bucket_name, file.filename)

    df = pd.read_csv(file.filename)
    df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes

    X = df
    prediction = model.predict(X)

    count = sum(prediction)
    co = len(prediction)
    p = count / co * 100 if co > 0 else 0

    return render_template('show.html', p=p, prediction_text='Predicted Successfully')
