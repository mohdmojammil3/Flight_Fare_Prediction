import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
import os
import joblib


def preprocess_data(train_data, test_data):
    # Define columns to transform
    numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()

    # Define preprocessing steps for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_encoder', OrdinalEncoder())
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ])

    # Fit and transform the data
    preprocessed_train_data = preprocessor.fit_transform(train_data)
    preprocessed_test_data = preprocessor.transform(test_data)

    # Get feature names for numeric and categorical data
    preprocessed_column_names = numeric_features + categorical_features

    # Convert the processed arrays back to DataFrames
    preprocessed_train_df = pd.DataFrame(preprocessed_train_data, columns=preprocessed_column_names)
    preprocessed_test_df = pd.DataFrame(preprocessed_test_data, columns=preprocessed_column_names)

    return preprocessed_train_df, preprocessed_test_df, preprocessor

def save_processed_data(train_data, test_data, train_file_path, test_file_path):
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)

if __name__ == "__main__":
    # Read data
    tndf = pd.read_csv(r'F:\working project\Flight_Fare_Prediction\data\traindata\train_data.csv')
    train_data  = tndf[['Total_Stops', 'journey_Month', 'Dep_hour', 'Arrival_hour','Duration_hours', 'Airline']]

    tedf = pd.read_csv(r"F:\working project\Flight_Fare_Prediction\data\testdata\test_data.csv")
    test_data = tedf[['Total_Stops', 'journey_Month', 'Dep_hour', 'Arrival_hour','Duration_hours', 'Airline']]

    preprocessed_train_data, preprocessed_test_data, preprocessor = preprocess_data(train_data, test_data)


    
    # Save the preprocessed data
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    save_processed_data(preprocessed_train_data, preprocessed_test_data, os.path.join("data", "processed", "processed_train.csv"), os.path.join("data", "processed", "processed_test.csv"))
    
    processed_train_file_path = os.path.join("data", "processed", "processed_train.csv")
    print(f"processed_train.csv is saved to : {processed_train_file_path}")
    
    processed_test_file_path =os.path.join('data','processed','processed_test.csv')
    print(f"processed_test.csv is saved to : {processed_test_file_path}")
    
    # Save the preprocessor as a .pkl file
    preprocessor_file_path = os.path.join("models", "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_file_path)
    print(f"Preprocessor.pkl is saved to : {preprocessor_file_path}")
