import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from data_transformation import DataTransformation
from model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass #without using __init__ we can intialize our class variable
class DataIngestionConfig:
    train_data_path : str=os.path.join("models","train.csv")
    test_data_path : str=os.path.join("models","test.csv")
    raw_data_path : str=os.path.join("models","raw.csv")

##Reading the data from the source and return to where does the data is needed 
class DataIngestion:
    def __init__(self):
        self.ingestion_config =DataIngestionConfig()
        
    def initate_data_ingestion(self):
        logging.info('Enterd into the data ingestion method')
        try:
            df=pd.read_csv('data/zomato.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Saved the raw data in the models folder')


            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)


            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Ingestion of the data has been completed ')



            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except CustomException as e:
            raise CustomException(e,sys)


if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initate_data_ingestion()

    data_tranformation = DataTransformation()
    train_arr,test_arr= data_tranformation.initiate_data_transformation(train_data,test_data)
    modeltrianer = ModelTrainer()
    print(modeltrianer.intiate_model_trainer(train_arr,test_arr))
