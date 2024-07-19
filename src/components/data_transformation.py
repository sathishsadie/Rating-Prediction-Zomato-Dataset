import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin, BaseEstimator
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('models', 'preprocessor.pkl')


class DenseTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns = ['votes', 'approx_cost(for two people)']
            categorical_columns = ['online_order', 'book_table', 'rest_type', 'cuisines', 'listed_in(type)',
                                   'listed_in(city)']
            # pca_cols = 'dish_liked'

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=413733)),
                    ('scaler', StandardScaler())
                ]
            )

            # svd_pca_pipeline = Pipeline(
            #     steps=[
            #         ('tfidf', TfidfVectorizer()),
            #         ('to_dense', DenseTransformer()),
            #         ('pca', PCA(n_components=1))
            #     ]
            # )

            logging.info('Numerical, Categorical, and PCA Columns are transformed and standardized.')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)#,
                    # ('pca_pipeline', svd_pca_pipeline, pca_cols)
                ],remainder='drop'
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read the train and test data')
            logging.info('Obtaining preprocessing object.')

            preprocessing_obj = self.get_data_transformation_object()
            target_column = 'rate'

            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info('Applying data transformation to the objects.')
            preprocessing_obj.fit(input_feature_train_df)
            save_object(
                        filepath=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
                        )
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info('Data transformation has been completed.')

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            # Save the preprocessing object
            # joblib.dump(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file_path, protocol=5)
            logging.info(f"Preprocessing object saved to {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)
