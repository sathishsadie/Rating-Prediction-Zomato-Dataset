from dataclasses import dataclass
import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('models','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try :
            numerical_columns = ['votes','approx_cost(for two people)']
            categorical_columns = ['online_order','book_table','rest_type','cuisines','listed_in(type)','listed_in(city)']
            pca_cols = ['dish_liked']
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('label_encoder',LabelEncoder()),
                    ('scaler',StandardScaler())
                ]
            )
            pca_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer()), 
                    ('pca', PCA(n_components=1))
            ])
            logging.info('Numerical and Categorical Columns and PCA are converted into the needed format and standardized it .')


            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns),
                    ('pca_pipeline',pca_pipeline,pca_cols)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def intiate_data_transformation(self,train_path,test_path):
        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read the train and test data')
            logging.info('Obtaining preprocessing object.')


            preprocessing_obj = self.get_data_transformation_object()
            target_column = 'rate'
            

            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]
            logging.info('Applying data transformation  to the objects.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info('Data transformation has been completed .')
            train_arr = np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df]

            save_object(
                filepath=self.data_transformation_config.preprecessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprecessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)