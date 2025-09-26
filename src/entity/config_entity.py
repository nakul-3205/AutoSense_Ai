from datetime import datetime
import os
from src import constant

class TrainingPipelineConfig:
    def __init__ (self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=constant.PIPELINE_NAME
        self.artifact_name=constant.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.model_dir=os.path.join("final_model")
        self.timestamp: str=timestamp


class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir=os.path.join(
            training_pipeline_config.artifact_dir,constant.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, constant.DATA_INGESTION_FEATURE_STORE_DIR_NAME, constant.FILE_NAME
            )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, constant.DATA_INGESTION_INGESTED_DIR, constant.TRAIN_FILE_NAME
            )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, constant.DATA_INGESTION_INGESTED_DIR, constant.TEST_FILE_NAME
            )
        self.train_test_split_ratio: float = constant.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name: str = constant.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = constant.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join( training_pipeline_config.artifact_dir, constant.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, constant.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, constant.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, constant.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, constant.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, constant.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, constant.TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            constant.DATA_VALIDATION_DRIFT_REPORT_DIR,
            constant.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )


class DataTransformationConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,constant.DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,constant.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            constant.TRAIN_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  constant.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            constant.TEST_FILE_NAME.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, constant.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            constant.PREPROCESSING_OBJECT_FILE_NAME,)

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, constant.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, constant.MODEL_TRAINER_TRAINED_MODEL_DIR,
            constant.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = constant.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = constant.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD


class DataUploadConfig:
    def __init__(self):
        self.raw_data_path =constant.RAW_DATA_PATH
        self.mongo_uri = os.getenv("MONGODB_URL")
        self.database_name = constant.DATA_INGESTION_DATABASE_NAME
        self.collection_name = constant.DATA_INGESTION_COLLECTION_NAME
