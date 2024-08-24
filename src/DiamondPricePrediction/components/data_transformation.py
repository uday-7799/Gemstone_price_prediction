import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.DiamondPricePrediction.exception import CustomException
from src.DiamondPricePrediction.logger import logging
import os

from src.DiamondPricePrediction.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):

        try:
            logging.info("data transformation initiated")

            # listing categorical and numerical columns seperately
            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
            
            # categories fo categorical columns
            cut_categories =  ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J'] 
            clarity_categories =  ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'] 

            logging.info("pipeline initiated")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ("scaler",StandardScaler())
                ]  
            )

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('cat_pipeline',cat_pipeline,categorical_columns),
                ('num_pipeline',num_pipeline,numerical_columns)
            ])


            return preprocessor

        except Exception as e:
            logging.info("exception occured in get_data_transformation")
            raise CustomException(e,sys)        


    def initiate_data_transformation(self,train_path,test_path):

        try:
        
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read the train and test data")

            # creating an object for preprocessor
            logging.info("obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformation()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("applying preprocessor object on train dataframe and testinf dataframe")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            # addind both input_feature_arr and target_feature_arr 
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation method")
            raise CustomException(e,sys)

