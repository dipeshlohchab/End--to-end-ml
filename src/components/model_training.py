import os, sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from src.utils import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    train_model_file_path= os.path.join("artifacts/Model_Training", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting our data into dependent and independent features")
            X_Train, y_train, X_Test, y_test=(
                train_arr[:, :-1], 
                train_arr[:, -1], 
                test_arr[:, :-1], 
                test_arr[:, -1]
            )

            models= {
                "LogisticRegression": LogisticRegression(),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier()    
            }

            params = {
                "RandomForest": {
                    "class_weight":["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },
                "DecisionTree": {
                    "class_weight": ["balanced"],
                    "criterion": ['gini', "entropy", "log_loss"],
                    "splitter": ['best', 'random'],
                    "max_depth": [3, 4, 5, 6],
                    "min_samples_split": [2, 3, 4, 5],
                    "min_samples_leaf": [1, 2, 3],
                    "max_features": ["sqrt", "log2"]  # Removed "auto"
                },
                "LogisticRegression": {
                    "class_weight": ["balanced"],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga', 'lbfgs'],
                    'max_iter': [100, 200, 500, 1000]
                }
            }

            model_report:dict = evaluate_model(X_Train, y_train, X_Test, y_test, models, params)


            # Get Best Model
            best_model_score= max(model_report.values())
            best_model_name= list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model= models[best_model_name]
            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model: {best_model_name}, accuracy: {best_model_score}")

            save_object(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)


        except Exception as e:
            logging.error(f"Error in Model Training: {e}")
            raise CustomException(e)