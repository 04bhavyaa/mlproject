import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host=os.getenv('host')
user=os.getenv('user')
password=os.getenv('password')
database=os.getenv('db')

def read_sql_data():
    logging.info("Reading from MySQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=database)
        logging.info("Connection established", mydb)
        df= pd.read_sql("SELECT * FROM students", con=mydb)
        print(df.head())

        return df
    
    except Exception as e:
        raise CustomException(e,sys)
    