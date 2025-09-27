import os
import sys
import joblib
import mlflow
import mlflow.sklearn
import dagshub
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from src.utils.exception import CustomException
from src.utils.log_config import logger
from src.entity.artifact_entity import ModelTrainerArtifact, RegressionMetricArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfig

dagshub.init(repo_owner='nakul-3205', repo_name='AutoSense_Ai', mlflow=True)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig=None, data_transformation_artifact: DataTransformationArtifact=None):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact

    @staticmethod
    def evaluate_model(model, X, y) -> RegressionMetricArtifact:
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        # rmse = mean_squared_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        return RegressionMetricArtifact(mae=mae, rmse=rmse, r2=r2)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso()
            }

            best_r2 = -float("inf")
            best_model = None
            best_train_metrics = None
            best_test_metrics = None
            best_model_name = ""

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                train_metrics = self.evaluate_model(model, X_train, y_train)
                test_metrics = self.evaluate_model(model, X_test, y_test)

                # Log to MLflow
                with mlflow.start_run(run_name=model_name):
                    mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
                    mlflow.log_metrics({
                        "train_mae": train_metrics.mae,
                        "train_rmse": train_metrics.rmse,
                        "train_r2": train_metrics.r2,
                        "test_mae": test_metrics.mae,
                        "test_rmse": test_metrics.rmse,
                        "test_r2": test_metrics.r2
                    })
                    mlflow.sklearn.log_model(model, "model")

                # Update best model
                if test_metrics.r2 > best_r2:
                    best_r2 = test_metrics.r2
                    best_model = model
                    best_train_metrics = train_metrics
                    best_test_metrics = test_metrics
                    best_model_name = model_name

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.config.trained_model_file_path)
            logger.info(f"Best Model: {best_model_name} | R2: {best_r2}")

            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_file_path,
                train_metric_artifact=best_train_metrics,
                test_metric_artifact=best_test_metrics
            )

        except Exception as e:
            logger.error("Error during model training")
            raise CustomException(e, sys)
