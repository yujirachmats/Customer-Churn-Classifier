from flask import Flask, render_template, request
import pandas as pd
import joblib
from transformData import trans

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('CustomerHome.html')

@app.route('/Home', methods=['GET','POST'])
def customer():
    if request.method == 'POST':
        data = request.form
        df = pd.DataFrame(
            [[data['Gender'],data['SeniorCitizen'],data['Partner'],data['Dependents'],int(data['Tenure']),
            data['PhoneService'],data['MultipleLines'],data['InternetService'],data['OnlineSecurity'],
            data['OnlineBackup'],data['DeviceProtection'],data['TechSupport'],data['StreamingTV'],
            data['StreamingMovies'],data['Contract'],data['PaperlessBilling'],data['PaymentMethod'],
            float(data['MonthlyCharges']),float(data['TotalCharges'])]],columns=customer.columns[1:20])
        output = trans(df)
        # print(output)
        outcome = f'{round(model.predict_proba(output)[0][1]*100,2)}%'
        return render_template('Prediction.html',x=outcome)
        
if __name__ == '__main__':
    model = joblib.load('FinalModeljoblib')
    customer = joblib.load('DFjoblib')
    app.run(debug = True,host='localhost',port=5000)