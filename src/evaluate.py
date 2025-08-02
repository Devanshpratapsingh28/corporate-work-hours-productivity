import pandas as pd
import joblib
import yaml
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import mlflow

params = yaml.safe_load(open('params.yaml'))
params_target = yaml.safe_load(open('params.yaml'))['common']['target']
params = params['evaluate']

def evaluate(data_path,model_path,target):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[params_target])
    y = df[params_target]

    model = joblib.load(open(model_path,'rb'))

    y_pred = model.predict(X)
    mae = mean_absolute_error(y,y_pred)
    rmse = root_mean_squared_error(y,y_pred)
    r2 = r2_score(y,y_pred)
    
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2 Score: {r2:.3f}")

if __name__ == "__main__":
    evaluate(params['data_path'],params['model_path'],params_target)    
