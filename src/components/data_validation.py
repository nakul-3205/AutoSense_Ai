import os
import sys
from dataclasses import dataclass
import pandas as pd
from scipy.stats import ks_2samp
from src.utils.exception import CustomException
from src.utils.log_config import logger
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.utils.main_utils import read_yaml_file, write_yaml_file
from src.constant import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self.schema_config['columns']  # get list of columns
            logger.info(f"Expected columns: {len(expected_columns)}, Dataframe columns: {len(dataframe.columns)}")
            missing_cols = [col for col in expected_columns if col not in dataframe.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
            return len(missing_cols) == 0
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                ks_test = ks_2samp(d1, d2)
                drift_detected = ks_test.pvalue < threshold
                if drift_detected:
                    status = False
                report[column] = {"p_value": float(ks_test.pvalue), "drift_status": drift_detected}

            drift_report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_path, content=report)
            return status

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Column validation
            if not self.validate_number_of_columns(train_df):
                logger.warning("Train dataframe does not contain all required columns")
            if not self.validate_number_of_columns(test_df):
                logger.warning("Test dataframe does not contain all required columns")

            # Dataset drift detection
            drift_status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            # Save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            return DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
