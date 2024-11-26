import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
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
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
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
                "ElasticNet": {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9],
                    'fit_intercept': [True, False]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.6, 0.75, 0.9],
                    'max_depth': [3, 5, 7]
                },
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'reg_alpha': [0, 1, 10],
                    'reg_lambda': [0, 1, 10]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200],
                    'l2_leaf_reg': [1, 3, 5]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'loss': ['linear', 'square', 'exponential']
                }
            }

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            
            # To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            model_names = list(params.keys())
            actual_model = ""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]
            mlflow.set_registry_uri("https://dagshub.com/04bhavyaa/mlproject.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # MLFlow
            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)
                mlflow.log_params(best_params)
                mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name="best_model")
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            mae = mean_absolute_error(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            r2 = r2_score(y_test, predicted)

            # Return the model and metrics
            print(f'"best_model": {best_model}, "model_name": {best_model_name}, "mae": {mae}, "mse": {mse}, "r2": {r2}')

        except CustomException as e:
            raise CustomException(e, sys)
