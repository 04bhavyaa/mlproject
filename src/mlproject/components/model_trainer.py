import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_model

import dagshub
dagshub.init(repo_owner='04bhavyaa', repo_name='mlproject', mlflow=True)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split the data into train and test arrays")
            X_train, y_train, X_test, y_test = [
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            ]
            
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
            }
            
            # Hyperparameter grids for GridSearchCV
            param_grids = {
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                },
                "Ridge Regression": {
                    'alpha': [0.1, 1.0, 10.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                    'fit_intercept': [True, False]
                },
                "Lasso Regression": {
                    'alpha': [0.1, 1.0, 10.0],
                    'fit_intercept': [True, False],
                    'positive': [True, False],
                    'selection': ['cyclic', 'random']
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'reg_alpha': [0, 1, 10],
                    'reg_lambda': [0, 1, 10]
                }
            }

            best_model_score = -float('inf')
            best_model = None
            best_params = None

            # MLflow logging setup
            mlflow.set_registry_uri("https://dagshub.com/04bhavyaa/mlproject.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Use GridSearchCV for hyperparameter tuning
            with mlflow.start_run():
                for model_name, model in models.items():
                    logging.info(f"Training {model_name}")

                    # Perform GridSearchCV to find the best hyperparameters
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
                    grid_search.fit(X_train, y_train)

                    # Get the best model and parameters
                    best_params_for_model = grid_search.best_params_
                    best_model_for_model = grid_search.best_estimator_

                    # Evaluate the model
                    predicted_qualities = best_model_for_model.predict(X_test)
                    rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)

                    # Check and log parameters
                    try:
                        mlflow.log_params(best_params_for_model)
                    except Exception as e:
                        logging.error(f"Error logging params: {e}")
                    
                    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
                    mlflow.sklearn.log_model(model, model_name)

                    # Check if this model is the best performing one
                    if rmse < best_model_score or best_model_score == -float('inf'):
                        best_model_score = rmse
                        best_model = best_model_for_model
                        best_params = best_params_for_model

                    # Register the model in MLFlow Model Registry
                    if tracking_url_type_store != "file":
                        mlflow.sklearn.log_model(best_model_for_model, model_name, registered_model_name=model_name, input_example=X_test[:5])  # Providing input example
                    else:
                        mlflow.sklearn.log_model(best_model_for_model, model_name)

            if best_model_score == -float('inf'):
                raise CustomException("No best model found")
            
            logging.info(f"Best model: {best_model} with RMSE: {best_model_score}")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Return final metrics and best model
            predicted = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            r2 = r2_score(y_test, predicted)

            print(f'"best_model": {best_model}, "mae": {mae}, "mse": {mse}, "r2": {r2}')

        except CustomException as e:
            raise CustomException(e, sys)