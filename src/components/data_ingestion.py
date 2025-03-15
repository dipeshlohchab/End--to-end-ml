import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path= os.path.join("artifacts/data_ingestion", 'train.csv')
    test_data_path= os.path.join("artifacts/data_ingestion", 'test.csv')
    raw_data_path= os.path.join("artifacts/data_ingestion", 'raw.csv')

# notebook\data\income_data.csv

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiateDataIngestion(self):
        logging.info(f"Initiating Data Ingestion")
        try:
            logging.info(f"Data Reading using Pandas")
            data= pd.read_csv(os.path.join('notebook/data', 'cleaned_data.csv'))
            logging.info(f"Data Ingested")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            train_set, test_set= train_test_split(data, test_size=0.3, random_state=42) 
            logging.info(f"Data Splitted into Train and Test")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Data Ingestion Successful")
            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.error(f"Error in Data Ingestion: {str(e)}")
            raise CustomException(e)

if __name__ == "__main__":
    obj= DataIngestion()
    train_path, test_path= obj.initiateDataIngestion()

    data_transformation_obj= DataTransformation()
    train_arr, test_arr, _= data_transformation_obj.initiateDataTransformation(train_path, test_path)

    # src\components\data_ingestion.py