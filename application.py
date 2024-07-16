from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

application = Flask(__name__)

app = application

# Load the preprocessor and model
preprocessor_file_path = os.path.join("models", "preprocessor.pkl")
model_file_path = os.path.join("models", "model.pkl")
preprocessor = joblib.load(preprocessor_file_path)
model = joblib.load(model_file_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = {
        'Total_Stops': [int(request.form['Total_Stops'])],
        'Airline': [request.form['Airline']],
        'journey_Month': [int(request.form['journey_Month'])],
        'Dep_hour': [int(request.form['Dep_hour'])],
        'Arrival_hour': [int(request.form['Arrival_hour'])],
        'Duration_hours': [int(request.form['Duration_hours'])]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Preprocess the new data
    preprocessed_data = preprocessor.transform(input_df)
    
    # Make predictions
    predictions = model.predict(preprocessed_data)
    
    # Return the result
    return jsonify({'prediction': predictions[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0")
