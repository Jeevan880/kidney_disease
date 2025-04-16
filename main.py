from kidney_disease import logger

from kidney_disease.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from kidney_disease.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from kidney_disease.pipeline.stage_03_model_training import ModelTrainingPipeline
from kidney_disease.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline



STAGE_NAME="Data Ingestion stage"

if __name__=="__main__":
    try:
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<")
        obj=DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Completed <<<<<<<<<")
    except Exception as e:
        raise e


STAGE_NAME = "Prepare Base Model"


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<")
        obj=PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Completed <<<<<<<<<")
    except Exception as e:
        raise e
    
STAGE_NAME = "Model Training"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<")
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Completed <<<<<<<<<")
    except Exception as e:
        raise e


STAGE_NAME = "Model Evaluation"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<")
        obj=ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Completed <<<<<<<<<")
    except Exception as e:
        raise e

