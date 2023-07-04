from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import mysql.connector

app = Flask(__name__)

loaded_model = pd.read_pickle(open('randomforest_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/user_value', methods=['POST'])
def userdetails():
    user_name = request.form['user_name']
    return render_template('index.html', user=user_name)


@app.route('/services')
def service():
    return render_template('services.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    # Retrieve input values from the form
    gender = int(request.form['gender'])
    married = int(request.form['married'])
    dependents = int(request.form['dependents'])
    education = int(request.form['education'])
    self_employed = int(request.form['self_employed'])
    applicant_income = float(request.form['applicant_income'])
    coapplicant_income = float(request.form['coapplicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_term = int(request.form['loan_term'])
    credit_history_available = int(request.form['credit_history_available'])
    housing = int(request.form['housing'])
    locality = int(request.form['locality'])

    # Create the input DataFrame
    input_data = {
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Term': [loan_term],
        'Credit_History_Available': [credit_history_available],
        'Housing': [housing],
        'Locality': [locality]
    }
    input_df = pd.DataFrame(input_data)

    predicted_risk = loaded_model.predict(input_df)
    if predicted_risk==1:
        result="There is a risk(1) in the applicant profile"
    else:
        result = "There is a No risk(0) in the applicant profile"
    result_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Term': loan_term,
        'Credit_History_Available': credit_history_available,
        'Housing': housing,
        'Locality': locality,
        'Result': result
    }

    return render_template('services.html', result=result_data)


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
