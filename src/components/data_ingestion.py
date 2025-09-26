import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import numpy as np
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.utils.exception import CustomException
from src.utils.logging import logging
from dotenv import load_dotenv
import sys

load_dotenv()

MONGO_URL=os.getenv('MONGODB_URL')

class DataIngestion:
    def __init__(self):
        try:
            config=DataIngestionConfig()
            self.config=config
            self.required_columns = [
            "make",
            "mileage",
            "year",
            "engine_hp",
            "vehicle_age",
            "transmission",
            "fuel_type",
            "drivetrain",
            "body_type"
        ]
        except Exception as e:
            raise CustomException(e,sys)

    def load_data(self):
        try:
            database_name=self.config.database_name
            collection_name=self.config.collection_name
            self.mongo_client=MongoClient(MONGO_URL)
            collection=self.mongo_client[database_name][collection_name]
            df= pd.DataFrame(list(collection.find()))
            if '_id' in df.columns.to_list():
                df=df.drop(columns=['_id'],axis=1)
            df.replace({"na":np.nan},inplace=True)
            print(df.head())
            logging.info('downloaded from Mongo')

            df = df[[col for col in self.required_columns if col in df.columns]]
            print(df.head())
            logging.info('Dropped columns')
            return df


        except Exception as e:
            logging.error('Error at upload_data in DataIngestion Pipeline')
            raise CustomException(e,sys)

    def split_and_save_data(self, df):
        logging.info("Splitting data into train and test sets")
        train_df, test_df = train_test_split(df, test_size=self.config.test_size, random_state=42)
        os.makedirs(self.config.artifact_folder, exist_ok=True)
        train_path = os.path.join(self.config.artifact_folder, "train.csv")
        test_path = os.path.join(self.config.artifact_folder, "test.csv")
        raw_path = os.path.join(self.config.artifact_folder, "raw.csv")
        df.to_csv(raw_path, index=False)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        return DataIngestionArtifact(train_path=train_path, test_path=test_path, raw_path=raw_path)
    def run(self):
        try:
            df = self.load_data()
            artifact = self.split_and_save_data(df)
            logging.info("Data Ingestion completed successfully")
            return artifact
        except Exception as e:
            logging.error("Error occurred in Data Ingestion Pipeline")
            raise CustomException(e, sys)

