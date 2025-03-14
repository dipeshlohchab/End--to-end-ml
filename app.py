from flask import Flask
from src.logger import logging
from src.exception import CustomException
import sys, os
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        raise Exception('This is a test exception')        
    except Exception as e:
        abc= CustomException(e)
        logging.info(abc.error_message)
        return 'This is End to End ML Project'
    
if __name__ == '__main__':
    app.run(debug=True)
