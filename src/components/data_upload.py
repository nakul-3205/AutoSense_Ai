from src.utils.exception import CustomException
from src.utils.logging import logging
from pymongo import MongoClient
from dotenv import load_dotenv
from src.entity.config_entity import DataUploadConfig
import os
import pandas as pd
import sys

load_dotenv()
MONGO_URL = os.getenv('MONGODB_URL')


class RawUploadPipeline:
    def __init__(self):
        self.config = DataUploadConfig()
        self.mongo_url = MONGO_URL
        self.mongo_client = MongoClient(self.mongo_url)
        db_name = self.config.database_name
        collection_name = self.config.collection_name
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]

    def upload_raw_csv(self):
        try:
            raw_file_path = self.config.raw_data_path
            logging.info(f"Uploading raw CSV from {raw_file_path} to MongoDB")
            df = pd.read_csv(raw_file_path)
            drop_cols = ["Unnamed: 0", "index"]
            df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
            df.reset_index(drop=True, inplace=True)
            json_records = df.to_dict(orient="records")
            self.collection.insert_many(json_records)
            logging.info(f"Successfully uploaded {len(json_records)} records to MongoDB")
        except Exception as e:
            logging.error("Error occurred while uploading data to MongoDB")
            raise CustomException(e, sys)

    def run(self):
        self.upload_raw_csv()


if __name__ == "__main__":
    pipeline = RawUploadPipeline()
    pipeline.run()
