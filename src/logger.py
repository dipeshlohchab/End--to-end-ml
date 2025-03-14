import os
import logging
from datetime import datetime

# Create log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.log"

# Define log directory
LOG_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Full log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.DEBUG,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s'
)

# if __name__ == '__main__':
#     logging.info('This is a test log message')
#     logging.error('This is an error message')
