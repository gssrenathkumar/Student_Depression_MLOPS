from studentdepression.componenets.data_ingestion import DataIngestion
from studentdepression.componenets.data_transformation import DataTransformation
from studentdepression.componenets.data_validation import DataValidation
from studentdepression.componenets.model_trainer import ModelTrainer
from studentdepression.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelTrainerConfig, TrainingPipelineConfig
from studentdepression.exception.exception import NetworkSecurityException
from studentdepression.logging.logger import logging
import sys

if __name__=='__main__':
    try:
        logging.info("Initialise the program")
        trainingpipelineconfig=TrainingPipelineConfig()

        logging.info("Starting Phase 1 - Data Ingestion")
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Completed Phase 1 - Data Ingestion")
        print(dataingestionartifact)

        logging.info("Starting Phase 2 - Data Validation")
        datavalidationconfig = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,datavalidationconfig)
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Completed Phase 2 - Data Validation")
        print(data_validation_artifact)

        logging.info("Starting Phase 3 - Data Transformation")
        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Completed Phase 3 - Data Transformation")
        print(data_transformation_artifact)

        logging.info("Starting Phase 4 - Model Training")
        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(data_transformation_artifact,model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Completed Phase 4 - Model Trainer")
        print(model_trainer_artifact)





    except Exception as e:
           raise NetworkSecurityException(e,sys)