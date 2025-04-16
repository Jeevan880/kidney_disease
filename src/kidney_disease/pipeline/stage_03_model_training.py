from kidney_disease.config.configuration import ConfigurationManager
from kidney_disease import logger
from kidney_disease.components.model_training import ModelTraining

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = ModelTraining(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train_model()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<")
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Completed <<<<<<<<<")
    except Exception as e:
        raise e