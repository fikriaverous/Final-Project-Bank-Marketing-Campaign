from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

app = Flask(__name__)

df = pd.read_csv('bank.csv')
df_raw = df.drop(['pdays', 'poutcome'], axis=1)

def load_le_default():
    global le_default
    with open('le_default.pkl','rb') as ledefault:
        le_default = pickle.load(ledefault)

def load_le_housing():
    global le_housing
    with open('le_housing.pkl','rb') as lehousing:
        le_housing = pickle.load(lehousing)

def load_le_loan():
    global le_loan
    with open('le_loan.pkl','rb') as leloan:
        le_loan = pickle.load(leloan)

def load_ohe_test():
    global ohe_test
    with open('ohe_test.pkl','rb') as ohe:
        ohe_test = pickle.load(ohe)

def load_model():
    global model
    with open('rfc_best.pkl','rb') as mymodel:
        model = pickle.load(mymodel)

@app.route('/', methods=['POST', 'GET'])
def dashboard():
    job = df_raw['job'].unique()
    marital = df_raw['marital'].unique()
    education = df_raw['education'].unique()
    default = df_raw['default'].unique()
    housing = df_raw['housing'].unique()
    loan = df_raw['loan'].unique()
    contact = df_raw['contact'].unique()
    month = df_raw['month'].unique()

    if request.method == 'POST':
        body = request.form

        age = body['age']
        job = body['job']
        marital = body['marital']
        education = body['education']
        default = body['default']
        balance = body['balance']
        housing = body['housing']
        loan = body['loan']
        contact = body['contact']
        day = body['day']
        month = body['month']
        duration = body['duration']
        campaign = body['campaign']
        previous = body['previous'] 

        data = {'age':[age],
        'job':[job],
        'marital':[marital],
        'education':[education],
        'default': [default],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day': [day],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'previous': [previous]}

        dfinput = pd.DataFrame(data)

        ser = ohe_test.transform(dfinput)

        dfinput = pd.DataFrame(
            ser, columns=list(df_raw['job'].unique())+list(df_raw['marital'].unique())+list(df_raw['education'].unique())+list(df_raw['contact'].unique())+list(df_raw['month'].unique())+['age','default','balance','housing','loan','day','duration','campaign','previous']
        )

        dfinput['default'] = le_default.transform(dfinput['default'])
        dfinput['housing'] = le_housing.transform(dfinput['housing'])
        dfinput['loan'] = le_loan.transform(dfinput['loan'])

        for item in dfinput.drop(['default','housing','loan'], axis=1):
            dfinput[item] = dfinput[item].astype(int)

        dfinput.columns = ['admin.', 'technician', 'services', 'management', 'retired',
                            'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown_job',
                            'self-employed', 'student', 'married', 'single', 'divorced',
                            'secondary', 'tertiary', 'primary', 'unknown_education', 'unknown_contact', 'cellular',
                            'telephone', 'may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan',
                            'feb', 'mar', 'apr', 'sep', 'age', 'default', 'balance', 'housing',
                            'loan', 'day', 'duration', 'campaign', 'previous']

        dfinput.drop(['admin.', 'married', 'secondary', 'unknown_contact'], axis=1, inplace=True)

        hasil = model.predict(dfinput)
        if hasil == 1:
            hasil = 'Potential Client'
        else:
            hasil = 'Need a Future Campaign!'

        job1 = df_raw['job'].unique()
        marital1 = df_raw['marital'].unique()
        education1 = df_raw['education'].unique()
        default1 = df_raw['default'].unique()
        housing1 = df_raw['housing'].unique()
        loan1 = df_raw['loan'].unique()
        contact1 = df_raw['contact'].unique()
        month1 = df_raw['month'].unique()

        return render_template('predict.html', hasil=hasil, job=job1, marital=marital1, education=education1, default=default1, housing=housing1, loan=loan1, contact=contact1, month=month1)
    return render_template('predict.html', job=job, marital=marital, education=education, default=default, housing=housing, loan=loan, contact=contact, month=month)

if (__name__) == '__main__':
    load_le_default()
    load_le_housing()
    load_le_loan()
    load_model()
    load_ohe_test()
    app.run(
        debug=True,
        host='localhost',
        port=5000
    )