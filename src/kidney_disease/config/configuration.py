import os
from kidney_disease.constant import *
from kidney_disease.utils.common import create_directories,read_yaml
from kidney_disease.entity.config_entity import (DataIngestionConfig,PrepareBaseModelConfig,ModelTrainingConfig,EvaluationConfig)

class ConfigurationManager:
    def __init__(self,config_filepath=CONFIG_FILE_PATH,params_filepath=PARAMS_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self):
        config=self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
    
    def prepare_base_model_config(self)->PrepareBaseModelConfig:
        config=self.config.prepare_base_model

        create_directories([config.root_dir])

        base_model=PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )
        return base_model
    
    def get_training_config(self)->ModelTrainingConfig:
        training=self.config.training
        prepare_base_model=self.config.prepare_base_model
        training_data=os.path.join(self.config.data_ingestion.unzip_dir,"kidney-ct-scan-image")

        create_directories([
            Path(training.root_dir)
        ])

        return ModelTrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_image_size=self.params.IMAGE_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
        )
    
    def get_evaluation_config(self)->EvaluationConfig:
        eval_config=EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts\data_ingestion\kidney-ct-scan-image",
            all_params=self.params,
            mlflow_uri="https://dagshub.com/Jeevan880/kidney_disease.mlflow",
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
    
    
