from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        logging.error(f"Error in save_object: {str(e)}")
        raise CustomException(e)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        logging.info("Starting model evaluation...")
        report = {}

        # Apply SMOTE only to training data
        logging.info("Applying SMOTE to balance training data...")
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        logging.info(f"Training data after SMOTE: {X_train_smote.shape}, Labels: {y_train_smote.shape}")

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            param_grid = params[model_name]
            
            logging.info("Performing Grid Search for hyperparameter tuning...")
            GS = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            GS.fit(X_train_smote, y_train_smote)

            best_model = GS.best_estimator_
            logging.info(f"Best parameters for {model_name}: {GS.best_params_}")
            
            best_model.fit(X_train_smote, y_train_smote)
            y_pred = best_model.predict(X_test)

            report[model_name] = accuracy_score(y_test, y_pred),

            logging.info(f"Metrics for {model_name}: {accuracy_score(y_test, y_pred)}")

        logging.info("Model evaluation completed successfully.")
        return report

    except Exception as e:
        logging.error(f"Error in evaluate_model: {str(e)}")
        raise CustomException(e)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_objt:
            return pickle.load(file_objt)
    except Exception as e:
        raise CustomException(e)