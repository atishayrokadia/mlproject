import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationconfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initaite_data_ingestion(self):
        logging.info("data ingetion has started")
        try:
            print("-----------------1")
            df = pd.read_csv('notebook\data\stud.csv')
            print("-----------------2")
            logging.info('read the date set as df')
            print("-----------------3")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            print("------------------4")
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            print("------------------5")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            print("------------------6")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("data ingetion has completed")
            print("------------------7")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            return CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    print("HI")
    train_data,test_data = obj.initaite_data_ingestion()
    print("===================obtained train and test data==============")
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

