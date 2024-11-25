import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pymysql
import pickle
import numpy as np

load_dotenv()

host=os.getenv('host')
user=os.getenv('user')
password=os.getenv('password')
database=os.getenv('db')

def read_sql_data():
    logging.info("Reading from MySQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=database)
        logging.info("Connection established", mydb)
        df = pd.read_sql("SELECT * FROM students", con=mydb)
        print(df.head())

        return df
    
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try: 
        report = {}

        for i, model_name in enumerate(models.keys()):
            model = models[model_name]
            para = param.get(model_name, {})
            
            logging.info(f"Evaluating model: {model_name}")
            
            if not para:
                logging.warning(f"No parameters provided for {model_name}. Proceeding with default model.")
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} - Train R2: {train_model_score}, Test R2: {test_model_score}")
            report[model_name] = test_model_score

        if not report:
            logging.warning("Report is empty. No models evaluated successfully.")
        
        return report

    except Exception as e:
        logging.error("Error occurred in evaluate_model function", exc_info=True)
        raise CustomException(e, sys)
