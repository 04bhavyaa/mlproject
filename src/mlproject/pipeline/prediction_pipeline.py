import sys
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.utils import load_object
from src.mlproject.logger import logging

class PredictionPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        self.model = load_object(self.model_path)
        self.preprocessor = load_object(self.preprocessor_path)
    
    def encode_data(self, data):
        try:
            # Convert the data dictionary into a DataFrame with a single row
            data_df = pd.DataFrame([data])

            # Now apply the transformation
            transformed_data = self.preprocessor.transform(data_df)
            return transformed_data
        except Exception as e:
            raise CustomException(f"Error during data encoding: {str(e)}", sys)
    
    def predict(self, data):
        try:
            # Encode the data (transform it)
            transformed_data = self.encode_data(data)
            
            # Make prediction using the trained model
            prediction = self.model.predict(transformed_data)
            return prediction
        except Exception as e:
            raise CustomException(f"Error during prediction: {str(e)}", sys)
