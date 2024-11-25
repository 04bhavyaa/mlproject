import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object
from src.mlproject.components.model_trainer import ModelTrainer

class ModelMonitoring:
    def __init__(self):
        pass

    def evaluate_model_performance(self, X_test, y_test, model):
        try:
            predicted = model.predict(X_test)
            mae = mean_absolute_error(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            r2 = r2_score(y_test, predicted)
            logging.info(f"MAE: {mae}, MSE: {mse}, R2: {r2}")
            return mae, mse, r2
        except Exception as e:
            raise CustomException(e, sys)
