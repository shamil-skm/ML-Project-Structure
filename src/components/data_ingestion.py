import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import Datatransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts','train_data.csv')
    test_data_path: str= os.path.join('artifacts','test_data.csv')
    raw_data_path: str=os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
       self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok =True)
            logging.info('Created the train data directory')

            df.to_csv(self.ingestion_config.raw_data_path,index =False, header= True)
            logging.info('Saved the raw data to the respective path')

            train_set,test_set= train_test_split(df,test_size=0.2,random_state=42)
            logging.info('Split the dataset into train and test set')
        
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Saved the train and test set to the respective path')

            logging.info('Data ingestion process completed successfully')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__ == '__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    
    data_transformation=Datatransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))