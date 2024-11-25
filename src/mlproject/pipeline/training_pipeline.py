import sys
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object, read_sql_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.mlproject.logger import logging
import pandas as pd

class TrainingPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def preprocess_data(self, data: pd.DataFrame):
        try:
            # List of columns for preprocessing
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_features = ['math_score', 'reading_score']

            # Define transformations
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', StandardScaler())  # Replace with OneHotEncoder if needed
            ])

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # Combine transformers into a single column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_features),
                    ('num', numerical_transformer, numerical_features)
                ])

            return preprocessor

        except Exception as e:
            raise CustomException(f"Error during preprocessing: {str(e)}", sys)

    def train_and_save_model(self):
        try:
            # Step 1: Read data
            data = read_sql_data()
            logging.info("Data loaded successfully")

            # Step 2: Split data
            X = data.drop(columns=['math_score'])  # Features
            y = data['math_score']  # Target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Step 3: Preprocess data
            preprocessor = self.preprocess_data(X)

            # Step 4: Define and train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

            # Train the model
            pipeline.fit(X_train, y_train)
            logging.info("Model trained successfully")

            # Save the model and preprocessor
            save_object(self.model_path, pipeline.named_steps['model'])
            save_object(self.preprocessor_path, pipeline.named_steps['preprocessor'])

            logging.info(f"Model and preprocessor saved at {self.model_path} and {self.preprocessor_path}")

        except Exception as e:
            raise CustomException(f"Error during model training: {str(e)}", sys)
