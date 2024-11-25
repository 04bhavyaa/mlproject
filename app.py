import os
import sys
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.pipeline.prediction_pipeline import PredictionPipeline
import traceback

app = Flask(__name__)

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

        # Put the data into a list or dictionary that can be used for prediction
        data = {
            'gender': gender,
            'race_ethnicity': race_ethnicity,
            'parental_level_of_education': parental_level_of_education,
            'lunch': lunch,
            'test_preparation_course': test_preparation_course,
            'writing_score': writing_score,
            'reading_score': reading_score
        }

        # Call the prediction pipeline to predict math_score
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.predict(data)
        
        # Render the prediction result
        return render_template('index.html', predicted_score=prediction)

    except Exception as e:
        raise CustomException(e, sys)
        return render_template('index.html', prediction_text="Error occurred during prediction")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)