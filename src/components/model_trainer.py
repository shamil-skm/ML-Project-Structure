import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "trained_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "K Neighbors": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': [10, 50, 100, 200, 300]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                    'subsample': [0.5, 0.7, 1.0],
                    'n_estimators': [10, 50, 100, 200, 300]
                },
                "Linear Regression": {},
                "K Neighbors": {
                    'n_neighbors': [3, 5, 7, 9]
                },
                "XGBRegressor": {
                    'n_estimators': [10, 50, 100, 200, 300],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]
                },
                "CatBoosting Classifier": {
                    'depth': [3, 5, 7, 9],
                    'iterations': [10, 50, 100, 200, 300],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]
                },
                "AdaBoost": {
                    'n_estimators': [10, 50, 100, 200, 300],
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]
                }
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test,
                                                y_test=y_test, models=models, params=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info(f"Best model on both training and testing data: {best_model_name}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            model_r2 = r2_score(y_test, predicted)
            return model_r2

        except Exception as e:
            raise CustomException(e, sys)