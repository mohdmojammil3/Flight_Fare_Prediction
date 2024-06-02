import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(data, test_size=0.2):
    data = data[['Total_Stops', 'Price', 'journey_Month', 'Dep_hour', 'Arrival_hour', 'Duration_hours', 'Airline']]
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data

def save_data(train_data, test_data, train_file_path, test_file_path):
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)
    
if __name__ == "__main__":
    # Define file paths
    file_path_p = r"F:\working project\Flight_Fare_Prediction\data\processed\processed_data.csv"
    file_path_r = r"F:\working project\Flight_Fare_Prediction\data\rawdata\flight_fare_prediction.csv"
    
    # Load data
    raw_data_p = load_data(file_path_p)
    raw_data_r = load_data(file_path_r)
    
    # Select relevant columns and add the 'Airline' column
    rdp = raw_data_p[['Total_Stops', 'Price', 'journey_Month', 'Dep_hour', 'Arrival_hour', 'Duration_hours']]
    rdp.loc[:, 'Airline'] = raw_data_r['Airline']
    
    # Split the data
    train_data, test_data = split_data(rdp)
    
    # Define file paths for saving the split data
    train_file_path = r"F:\working project\Flight_Fare_Prediction\data\traindata\train_data.csv"
    test_file_path = r"F:\working project\Flight_Fare_Prediction\data\testdata\test_data.csv"
    
    # Save the split data
    save_data(train_data, test_data, train_file_path, test_file_path)