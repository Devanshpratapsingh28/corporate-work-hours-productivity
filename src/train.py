import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib
from urllib.parse import urlparse
from warnings import filterwarnings
filterwarnings("ignore")

params = yaml.safe_load(open('params.yaml'))
target_col = params['common']['target']
params = params['train']

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/devanshprataps6/corporate-work-hours-productivity.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "devanshprataps6"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "a16023baea3d90b6fd742c43657d87fa6d0fc8c9"

def hyperparameter_tuning(X,y,params):
    rf = RandomForestRegressor()
    rf_grid = RandomizedSearchCV(
        estimator=rf,
        param_distributions=params,
        n_jobs=-1,
        cv=5,
        n_iter=20,
        verbose=2
    )
    rf_grid.fit(X, y)
    return rf_grid


def train(data_path,model_path,param_grid,target):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    signature = infer_signature(X_train, y_train)

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with mlflow.start_run():

        random_search = hyperparameter_tuning(X_train,y_train,param_grid)
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        y_pred = best_model.predict(X_valid)
        mae = mean_absolute_error(y_valid,y_pred)
        rmse = root_mean_squared_error(y_valid,y_pred)
        r2 = r2_score(y_valid,y_pred)

        mlflow.log_params(best_params)

        mlflow.log_metric("Mean_Absolute_Error",mae)
        mlflow.log_metric("Root_Mean_Squared_Error",rmse)
        mlflow.log_metric("R2_Score",r2)

        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type!='file':
            mlflow.sklearn.log_model(best_model,"model")
        else:
            mlflow.sklearn.log_model(best_model, "model",signature=signature)

        # Create Directory to save model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        joblib.dump(best_model, model_path)

        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2 Score: {r2:.3f}")
        print(f"Model saved to {model_path}")



if __name__ == "__main__" :
    train(params['data_path'],params['model_path'],params['param_grid'],target_col)



