import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.exception import CustomException
from src.utils.log_config import logger  # fixed import


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact = None,
        data_transformation_config: DataTransformationConfig = None
    ):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

        # Define numeric and categorical columns
        self.categorical_cols = ['transmission', 'fuel_type', 'drivetrain', 'body_type','make']
        self.numeric_cols = [ 'mileage', 'year', 'engine_hp', 'vehicle_age']

    def get_transformer_object(self):
        try:
            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            scaler = StandardScaler()
            preprocessor = ColumnTransformer(
                transformers=[
                    ('ohe', ohe, self.categorical_cols),
                    ('scaler', scaler, self.numeric_cols)
                ],
                remainder='drop'
            )
            logger.info('Initialized StandardScaler and OneHotEncoder.')
            return preprocessor, scaler
        except Exception as e:
            logger.error('Error creating transformer object.')
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info('Starting data transformation process.')

            # Read validated train and test data
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            logger.info(f"Train data shape before transformation: {train_df.shape}")
            logger.info(f"Test data shape before transformation: {test_df.shape}")

            # Separate features and target
            target_column = "price"
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            X_train.fillna(0, inplace=True)
            X_test.fillna(0, inplace=True)

            preprocessing_obj, scaler_obj = self.get_transformer_object()
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)
            logger.info("Feature transformation completed.")

            # Combine features with target
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # Save transformed arrays
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)
            logger.info("Transformed train and test data saved as .npy files.")

            # Save preprocessing pipeline
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)
            with open(self.data_transformation_config.transformed_object_file_path, 'wb') as f:
                pickle.dump(preprocessing_obj, f)
            logger.info("Preprocessing pipeline saved successfully.")

            # Save scaler separately
            scaler_path = os.path.join(
                os.path.dirname(self.data_transformation_config.transformed_object_file_path),
                "scaler.pkl"
            )
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler_obj, f)
            logger.info("Scaler object saved successfully.")

            # Return artifact
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                scaler_object_file_path=scaler_path
            )

        except Exception as e:
            logger.error('Error in data transformation process.')
            raise CustomException(e, sys)
