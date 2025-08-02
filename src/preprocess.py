import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

params = yaml.safe_load(open('params.yaml'))['preprocess']
params_common = yaml.safe_load(open('params.yaml'))['common']

def preprocess(input_path,output_path,artfact_save_path,columns,target):
    df = pd.read_csv(input_path)
    cols = columns
    for i in df.columns :
       if i not in cols and i != target:
           df.drop(i, axis=1, inplace=True) 

    df['Job_Satisfaction'] = df['Job_Satisfaction'].astype('object')   

    X = df.drop(columns=['Productivity_Score'])
    y = df['Productivity_Score']

    # Pipeline Building
    num_pipeline = make_pipeline(
        StandardScaler()
    ) 

    cat_pipeline = make_pipeline(
        OneHotEncoder(handle_unknown='ignore')
    )

    # Applying num_pipeline to numerical features and cat_pipeline to categorical features and choosing column based on dtype.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
            ('cat', cat_pipeline, make_column_selector(dtype_include=object))
        ]
    )

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)     

    # Applying preprocessing pipelines
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    # Saving the preprocessed data as train.csv and test.csv
    os.makedirs(output_path, exist_ok=True)
    train_df = pd.DataFrame(X_train_pre,columns=preprocessor.get_feature_names_out())
    train_df[target] = y_train.values
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)

    test_df = pd.DataFrame(X_test_pre,columns=preprocessor.get_feature_names_out())
    test_df[target] = y_test.values
    test_df.to_csv(os.path.join(output_path,'test.csv'),index = False)

    # Saving Preprocessor pipeline as artifact
    os.makedirs(os.path.dirname(artfact_save_path), exist_ok=True)
    joblib.dump(preprocessor, artfact_save_path)

    print(f"Preprocessing complete. Train and test data saved to {output_path} folder.")

if __name__ == "__main__":
    preprocess(params['input'],params['output'],params['artifact_path'],params_common['columns'],params_common['target'])

    
