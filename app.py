import os
import sys
import numpy as np
import pickle
import mlflow
from flask import Flask, request, jsonify, render_template
from src.mlproject.logger import logging  # Import the logger
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.pipeline.prediction_pipeline import PredictionPipeline
import traceback

app = Flask(__name__)

# Initialize Dagshub (this automatically configures MLflow)
import dagshub
dagshub.init(repo_owner='04bhavyaa', repo_name='mlproject', mlflow=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the data from the form
        gender = request.form['gender']
        race_ethnicity = request.form['race_ethnicity']
        parental_level_of_education = request.form['parental_level_of_education']
        lunch = request.form['lunch']
        test_preparation_course = request.form['test_preparation_course']
        writing_score = float(request.form['writing_score'])
        reading_score = float(request.form['reading_score'])

        # Put the data into a dictionary for prediction
        data = {
            'gender': gender,
            'race_ethnicity': race_ethnicity,
            'parental_level_of_education': parental_level_of_education,
            'lunch': lunch,
            'test_preparation_course': test_preparation_course,
            'writing_score': writing_score,
            'reading_score': reading_score
        }

        # Log the data
        logging.info(f"Starting prediction for data: {data}")

        # Set up the MLflow experiment (create if not exists)
        experiment_name = 'math_score_prediction_experiment'
        mlflow.set_experiment(experiment_name)

        # Start an MLflow run for prediction
        with mlflow.start_run():
            # Log input data as parameters (optional)
            mlflow.log_param("gender", gender)
            mlflow.log_param("race_ethnicity", race_ethnicity)
            mlflow.log_param("parental_level_of_education", parental_level_of_education)
            mlflow.log_param("lunch", lunch)
            mlflow.log_param("test_preparation_course", test_preparation_course)
            mlflow.log_param("writing_score", writing_score)
            mlflow.log_param("reading_score", reading_score)

            # Call the prediction pipeline
            prediction_pipeline = PredictionPipeline()
            prediction = prediction_pipeline.predict(data)

            # Log the prediction result in MLflow
            mlflow.log_metric("predicted_math_score", prediction)

            # Log prediction in application logs
            logging.info(f"Prediction completed. Predicted math score: {prediction}")

            # Render the prediction result
            return render_template('index.html', predicted_score=prediction)

    except Exception as e:
        # Log error in application logs
        logging.error(f"Error occurred during prediction: {str(e)}")

        # Log the exception in MLflow (optional)
        with mlflow.start_run():
            mlflow.log_param("error", str(e))

        # Handle the exception properly
        return render_template('index.html', prediction_text="Error occurred during prediction")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

