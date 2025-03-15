from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
            
    except Exception as e:
        logging.error(f"Error in save_object: {str(e)}")
        raise CustomException(e)