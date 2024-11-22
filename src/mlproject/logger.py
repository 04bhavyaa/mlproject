import logging 
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH =os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    # level is set to INFO so that all the messages of level INFO and above are logged
    # filename is set to LOG_FILE_PATH so that the log is written to the file
    # format is set to include
    #   - date and time of the log
    #   - line number from where the log was called
    #   - name of the logger
    #   - level of the log
    #   - message of the log
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

