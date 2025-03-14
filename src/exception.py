import os
import sys
from src.logger import logging


def error_message_detailed(error, error_detailed):
    _, _, exc_tb = sys.exc_info()  
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in Python file [{file_name}] at line [{exc_tb.tb_lineno}]: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message_detailed(error_message, sys)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("An exception occurred: %s", str(e))
        raise CustomException(e)
