import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = joblib.load('DFjoblib')
dfChurn = joblib.load('dfchurnjoblib')

y_col = 'Churn'
cat_cols = df.nunique()[df.nunique()<=4].keys().tolist()
cat_cols = [x for x in cat_cols if x not in y_col]
num_cols = [x for x in df.columns if x not in [y_col] + ['customerID'] + cat_cols]

scaler = StandardScaler()
scaler.fit(df[num_cols])
one_hot_columns = dfChurn.columns[:36]

data = [['Female', 'No', 'Yes', 'No', 1, 'No', 'No', 'DSL','No', 'Yes', 'No', 'No', 'No', 'No', 'Month-to-month', 'Yes','Electronic check', 29.85, 29.85]]
dfx = pd.DataFrame(
    data,
    columns=df.columns[1:20])

def trans(data):
    dF = pd.DataFrame(data,index=[0])
    dF = pd.get_dummies(dF)
    dF = dF.reindex(columns=one_hot_columns, fill_value=0)
    ss = scaler.transform(data[num_cols])
    num = pd.DataFrame(ss,columns=num_cols)
    output = pd.concat([dF,num],axis='columns')
    return output

model = joblib.load('FinalModeljoblib')
print(model.predict_proba(trans(dfx))[0][1])
# print(model.score(trans(dfx)))
# print(df)