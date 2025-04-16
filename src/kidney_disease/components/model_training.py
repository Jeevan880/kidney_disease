
import tensorflow as tf
from pathlib import Path
from kidney_disease.config.configuration import ModelTrainingConfig

class ModelTraining:
    def __init__(self,config: ModelTrainingConfig):
        self.config=config
    
    def get_base_model(self):
        self.model=tf.keras.models.load_model(self.config.updated_base_model_path)
    
    def train_valid_generator(self):
        try:
            import PIL
            print("Pillow version:", PIL.__version__)
        except ImportError:
            print("Pillow is not installed.")
            
        datagenerator_kwargs=dict(
            rescale=1./255,
            validation_split=0.2
        )
        dataflow_kwargs=dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation='bilinear'
        )
        valid_datagenerator=tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator=valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset='validation',
            **dataflow_kwargs
        )
        if self.config.params_is_augmentation:
            train_datagenerator=tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                **datagenerator_kwargs
            )
            self.train_generator=train_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset='training',
                **dataflow_kwargs
            )
    @staticmethod
    def save_model(model:tf.keras.Model,path:Path):
        model.save(path)
        
    def train_model(self):
        self.steps_per_epoch=self.train_generator.samples//self.config.params_batch_size
        self.validation_steps=self.valid_generator.samples//self.config.params_batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps
        )

        self.save_model(model=self.model,path=self.config.trained_model_path)

