import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path= os.path.join("artifacts", 'train.csv')
    test_data_path= os.path.join("artifacts", 'test.csv')
    raw_data_path= os.path.join("artifacts", 'raw.csv')

# notebook\data\income_data.csv

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiateDataIngestion(self):
        logging.info(f"Initiating Data Ingestion")
        try:
            logging.info(f"Data Reading using Pandas")
            data= pd.read_csv(os.path.join('notebook/data', 'income_data.csv'))
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
    data_ingestion= DataIngestion()
    data_ingestion.initiateDataIngestion()

    # src\components\data_ingestion.py