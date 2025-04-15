import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from studentdepression.logging.logger import logging
from studentdepression.constants.training_pipeline import TARGET_COLUMN
from studentdepression.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from studentdepression.entity.config_entity import DataTransformationConfig
from studentdepression.exception.exception import NetworkSecurityException
from studentdepression.utils.common_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self, df: pd.DataFrame) -> ColumnTransformer:
        try:
            # Identify numerical and categorical columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if TARGET_COLUMN in numerical_cols:
                numerical_cols.remove(TARGET_COLUMN)

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_cols)
            ])

            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            preprocessor = self.get_data_transformer_object(train_df)
            preprocessor_obj = preprocessor.fit(input_feature_train_df)

            transformed_train = preprocessor_obj.transform(input_feature_train_df)
            transformed_test = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[transformed_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_test, np.array(target_feature_test_df)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj)
            save_object("final_model/preprocessor.pkl", preprocessor_obj)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
