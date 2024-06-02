# Flight Fare Prediction

This project is aimed at predicting flight fares based on several features such as total stops, airline, journey month, departure hour, arrival hour, and duration hours. The model utilized for prediction is RandomForestRegressor.

## Table of Contents
- [Installation]
- [Project Structure]
- [Usage]
- [Model Training]
- [Prediction]
- [References]
- [License]

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mohdmojammil3/Flight_Fare_Prediction.git
    cd flight-fare-prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project follows the following structure:

Dashboard/
│
├── data/
│ ├── traindata/
│ │ └── train_data.csv
│ ├── testdata/
│ │ └── test_data.csv
│ └── processed/
│ ├── processed_train_data.csv
│ └── processed_test_data.csv
│
├── models/
│ ├── preprocessor.pkl
│ └── model.pkl
│
├── notebooks/
│ ├── EDA.ipynb
│ ├── Model.ipynb
│ └── research.ipynb
│
├── src/
│ ├── components/
│ │ ├── data_preprocessing.py
│ │ ├── model_trainer.py
│ │ └── data_ingestion.py
│ └── pipeline/
│ └── predict_pipeline.py
│
├── static/
│ └── aircraft.jpg
│
├── templates/
│ └── index.html
│
├── setup.py
├── app.py
├── requirements.txt
└── README.md


## Usage

To run the application:

1. Ensure all dependencies are installed:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Flask application:
    ```bash
    python app.py
    ```

3. Open your web browser and go to `http://127.0.0.1:5000`.

## Model Training

The model is trained using the RandomForestRegressor algorithm. The training process includes the following steps:

1. Data Ingestion: Load and preprocess the training data.
2. Data Preprocessing: Handle missing values, encode categorical features, and scale numerical features.
3. Model Training: Train the RandomForestRegressor model using the preprocessed training data.

## Prediction

To make predictions using the trained model:

1. Load the preprocessor and model from their respective .pkl files.
2. Preprocess the input data using the preprocessor.
3. Use the model to predict flight fares based on the preprocessed data.

## References

- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/en/2.0.x/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


