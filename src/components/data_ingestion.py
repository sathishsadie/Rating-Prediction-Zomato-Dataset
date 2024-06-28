import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

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
            logging.info('Saved the raw data in the artifacts folder')


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

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initate_data_transformation(train_data,test_data)

    modeltrianer = ModelTrainer()
    print(modeltrianer.intiate_model_trainer(train_arr,test_arr))
## all the training code and ploting and accuracy or error here
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def intiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split the train and test input array')
            x_train,x_test,y_train,y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'XGBRegressor': XGBRegressor(),
                'SVR': SVR(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()}
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "SVR":{}
            }





            model_report : dict = evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,models = models,param=params)


            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]


            best_model = models[best_model_name]
            if best_model_score<0.6 :
                raise CustomException("No Best Model",sys)
            

            logging.info(f'Best model is {best_model_name} with score {best_model_score}')

            save_object(
                filepath = self.model_trainer_config.train_model_file_path,
                obj = best_model
            )
           
            predicted = best_model.predict(x_test)
            r2_square = r2_score(predicted,y_test)
            return r2_square
        


        except Exception as e:
            raise CustomException(e,sys)
        