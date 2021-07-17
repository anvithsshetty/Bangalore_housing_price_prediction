
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from joblib import dump, load
app = Flask(__name__)
model = load('pipe.joblib') 
data = pd.read_csv('locations.csv')

@app.route('/')
def hello_world():
    locations = data['location'].unique()
    return render_template("index.html",locations=locations)


@app.route('/predict',methods=['POST']) 
def predict():
    location=request.form.get('location')
    Bath=request.form.get('bath')
    bhk=request.form.get('BHK')
    sqft=request.form.get('sqft')
    df = pd.DataFrame([[location,Bath,bhk,sqft]],columns=['location','bath','size_new','sqft'])
    prediction=model.predict(df)[0]
    print(prediction)
    print(location,Bath,bhk,sqft)
    return str(prediction)


if __name__=="__main__":
    app.run(debug=True,port=7575)

    