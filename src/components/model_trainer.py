import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model(preprocessor, X_train, y_train):
    # Transform the data using the preprocessor
    X_train_processed = preprocessor.transform(X_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    return model

def save_model(model, model_file_path):
    joblib.dump(model, model_file_path)

if __name__ == "__main__":
    # Load the training data
    train_data = pd.read_csv(r'F:\working project\Flight_Fare_Prediction\data\traindata\train_data.csv')
    
    # Ensure 'Price' is included in the DataFrame
    train_data = train_data[['Total_Stops', 'Airline', 'journey_Month', 'Dep_hour', 'Arrival_hour', 'Duration_hours', 'Price']]
    
    # Separate features and target
    X_train = train_data[['Total_Stops', 'Airline', 'journey_Month', 'Dep_hour', 'Arrival_hour', 'Duration_hours']]
    y_train = train_data['Price']
    
    # Load the preprocessor from preprocessor.pkl file
    preprocessor_file_path = os.path.join("models", "preprocessor.pkl")
    preprocessor = joblib.load(preprocessor_file_path)
    print(f"Loaded preprocessor from: {preprocessor_file_path}")
    
    # Train the model
    trained_model = train_model(preprocessor, X_train, y_train)
    
    # Save the trained model
    model_file_path = os.path.join('models', 'model.pkl')
    save_model(trained_model, model_file_path)
    print(f'model.pkl file is saved to: {model_file_path}')
