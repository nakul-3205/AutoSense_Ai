import os
import sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src.utils.exception import CustomException
from src.utils.log_config import logger  # fixed import
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig

load_dotenv()

MONGO_URL = os.getenv('MONGODB_URL')


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = None):
        try:
            self.config = data_ingestion_config or DataIngestionConfig()

            # Columns to keep from MongoDB
            self.required_columns = [
                "make",
                "mileage",
                "engine_hp",
                "vehicle_age",
                "transmission",
                "fuel_type",
                "drivetrain",
                "body_type",
                "price",  # added missing comma in original list
            ]
        except Exception as e:
            raise CustomException(e, sys)

    def load_data_from_mongo(self) -> pd.DataFrame:
        try:
            client = MongoClient(MONGO_URL)
            collection = client[self.config.database_name][self.config.collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            logger.info("Data loaded from MongoDB successfully")

            # Keep only required columns
            df = df[[col for col in self.required_columns if col in df.columns]]
            logger.info("Filtered required columns")
            return df
        except Exception as e:
            logger.error("Error loading data from MongoDB")
            raise CustomException(e, sys)

    def save_feature_store(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            os.makedirs(os.path.dirname(self.config.feature_store_file_path), exist_ok=True)
            df.to_csv(self.config.feature_store_file_path, index=False, header=True)
            logger.info(f"Saved feature store at {self.config.feature_store_file_path}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def split_and_save_data(self, df: pd.DataFrame):
        try:
            if df.empty:
                raise CustomException("No data found. Cannot split empty dataset.", sys)

            train_set, test_set = train_test_split(df, test_size=self.config.train_test_split_ratio, random_state=42)
            logger.info("Performed train-test split")

            # Save training and testing datasets
            os.makedirs(os.path.dirname(self.config.training_file_path), exist_ok=True)
            train_set.to_csv(self.config.training_file_path, index=False, header=True)
            test_set.to_csv(self.config.testing_file_path, index=False, header=True)
            logger.info("Saved training and testing datasets")
        except Exception as e:
            raise CustomException(e, sys)

    def run(self) -> DataIngestionArtifact:
        try:
            df = self.load_data_from_mongo()
            df = self.save_feature_store(df)
            self.split_and_save_data(df)

            artifact = DataIngestionArtifact(
                trained_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path
            )
            logger.info("Data Ingestion completed successfully")
            return artifact
        except Exception as e:
            logger.error("Error in Data Ingestion pipeline")
            raise CustomException(e, sys)
