import os
import sys
from src.components.data_upload import RawUploadPipeline
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_validation import DataValidation, DataValidationConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_training import ModelTrainer, ModelTrainerConfig
from src.entity.config_entity import TrainingPipelineConfig
from src.utils.log_config import logging
from src.utils.exception import CustomException

if __name__ == '__main__':
    try:
        # Pipeline config
        training_pipeline_config = TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info('Data ingestion started')
        data_ingestion_artifact = data_ingestion.run()
        logging.info('Data ingestion completed')

        # Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info('Data validation started')
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info('Data validation completed')

        # Print artifacts
        print(data_ingestion_artifact)
        print(data_validation_artifact)

        # Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info('Data transformation started')
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info('Data transformation completed')
        print(data_transformation_artifact)

        # Model Training
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        logging.info('Model training started')
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info('Model training completed')
        print(model_trainer_artifact)

    except Exception as e:
        raise CustomException(e, sys)
