import os
from kidney_disease.utils.common import get_size
import gdown
import zipfile
from kidney_disease import logger
from kidney_disease.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config=config
    
    def download_data(self):

        try:
            source_url=self.config.source_URL
            local_data_file=self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)

            prefix_id=source_url.split('/')[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+prefix_id, local_data_file)

            logger.info(f"Downloaded data from {source_url} to {local_data_file}")
        except Exception as e:
            raise e
    
    def unzip_data(self):
        unzip_dir=self.config.unzip_dir
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_dir)

