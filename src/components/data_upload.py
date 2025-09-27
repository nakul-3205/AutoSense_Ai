import os
import sys
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

from src.utils.exception import CustomException
from src.utils.log_config import logger  # fixed import
from src.entity.config_entity import DataUploadConfig

load_dotenv()
MONGO_URL = os.getenv("MONGODB_URL")


class RawUploadPipeline:
    def __init__(self):
        try:
            self.config = DataUploadConfig()
            self.mongo_client = MongoClient(MONGO_URL)
            db_name = self.config.database_name
            collection_name = self.config.collection_name
            self.db = self.mongo_client[db_name]
            self.collection = self.db[collection_name]
            logger.info(f"Initialized MongoDB client for database: {db_name}, collection: {collection_name}")
        except Exception as e:
            logger.error("Error initializing RawUploadPipeline")
            raise CustomException(e, sys)

    def upload_raw_csv(self):
        try:
            raw_file_path = self.config.raw_data_path
            logger.info(f"Uploading raw CSV from {raw_file_path} to MongoDB")

            df = pd.read_csv(raw_file_path)

            # Drop unnecessary columns if present
            drop_cols = ["Unnamed: 0", "index"]
            df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
            df.reset_index(drop=True, inplace=True)

            # Convert to JSON records and insert
            json_records = df.to_dict(orient="records")
            self.collection.insert_many(json_records)
            logger.info(f"Successfully uploaded {len(json_records)} records to MongoDB")
        except Exception as e:
            logger.error("Error occurred while uploading data to MongoDB")
            raise CustomException(e, sys)

    def run(self):
        self.upload_raw_csv()


if __name__ == "__main__":
    pipeline = RawUploadPipeline()
    pipeline.run()
