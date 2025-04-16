from kidney_disease.config.configuration import ConfigurationManager
from kidney_disease import logger
from kidney_disease.components.model_evaluation_mlflow import Evaluation

STAGE_NAME = "Model Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        eval_config=config.get_evaluation_config()
        evaluation=Evaluation(eval_config)
        evaluation.evaluate_model()
        # evaluation.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<")
        obj=ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Completed <<<<<<<<<")
    except Exception as e:
        raise e

