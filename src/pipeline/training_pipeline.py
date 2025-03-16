import os, sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from dataclasses import dataclass

if __name__ == "__main__":
    try:
        obj= DataIngestion()
        train_data_path, test_data_path= obj.initiateDataIngestion()

        data_transformation_obj= DataTransformation()
        train_arr, test_arr, _= data_transformation_obj.initiateDataTransformation(train_data_path, test_data_path)

        model_training_obj= ModelTrainer()
        model_training_obj.initiate_model_training(train_arr, test_arr)


    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise CustomException(e)
    
    # src\pipeline\training_pipeline.py