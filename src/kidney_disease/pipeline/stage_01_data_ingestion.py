
from kidney_disease.config.configuration import ConfigurationManager
from kidney_disease import logger
from kidney_disease.components.data_ingestion import DataIngestion

STAGE_NAME="Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.unzip_data()

if __name__=="__main__":
    try:
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Started <<<<<<<<<")
        obj=DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>Stage {STAGE_NAME} Completed <<<<<<<<<")
    except Exception as e:
        raise e
