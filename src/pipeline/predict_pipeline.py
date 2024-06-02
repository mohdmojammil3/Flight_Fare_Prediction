import pandas as pd
import joblib
import os

def load_preprocessor(preprocessor_file_path):
    """
    Load the preprocessor from a .pkl file.
    
    Args:
        preprocessor_file_path (str): Path to the preprocessor .pkl file.
        
    Returns:
        preprocessor: The loaded preprocessor object.
    """
    return joblib.load(preprocessor_file_path)

def load_model(model_file_path):
    """
    Load the trained model from a .pkl file.
    
    Args:
        model_file_path (str): Path to the model .pkl file.
        
    Returns:
        model: The loaded model object.
    """
    return joblib.load(model_file_path)

def preprocess_data(preprocessor, data):
    """
    Preprocess the input data using the provided preprocessor.
    
    Args:
        preprocessor: The preprocessor object to use for transforming the data.
        data (pd.DataFrame): The input data to preprocess.
        
    Returns:
        np.ndarray: The preprocessed data.
    """
    return preprocessor.transform(data)

def predict(model, preprocessed_data):
    """
    Make predictions using the trained model on the preprocessed data.
    
    Args:
        model: The trained model object.
        preprocessed_data (np.ndarray): The preprocessed input data.
        
    Returns:
        np.ndarray: The predictions made by the model.
    """
    return model.predict(preprocessed_data)

if __name__ == "__main__":
    # Paths to the preprocessor and model files
    preprocessor_file_path = os.path.join("F:\working project\Flight_Fare_Prediction\models", "preprocessor.pkl")
    model_file_path = os.path.join("F:\working project\Flight_Fare_Prediction\models", "model.pkl")
    
    # Load the preprocessor and model
    preprocessor = load_preprocessor(preprocessor_file_path)
    model = load_model(model_file_path)
    
    print(f"Loaded preprocessor from: {preprocessor_file_path}")
    print(f"Loaded model from: {model_file_path}")

    # Load new data for prediction
    
    input_data = pd.DataFrame({
        'Total_Stops': [2],
        'Airline': ['IndiGo'],
        'journey_Month': [6],
        'Dep_hour': [7],
        'Arrival_hour':[8],
        'Duration_hours':[1]
        
    })
    
    
    # Preprocess the new data
    preprocessed_data = preprocess_data(preprocessor, input_data)
    
    # Make predictions
    predictions = predict(model, preprocessed_data)
    
    # Print the predictions
    print("Predicted Price:", predictions)
