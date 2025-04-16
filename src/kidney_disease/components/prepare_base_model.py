
import tensorflow as tf
from pathlib import Path
from kidney_disease.config.configuration import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self,config: PrepareBaseModelConfig):
        self.config=config

    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        model.save(path)
    def get_base_model(self):
        self.model=tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights
        )
        self.save_model(path=self.config.base_model_path,model=self.model)
    
    def _prepare_base_model(self,free_all,freeze_till,model,learning_rate,classes):
        if free_all:
            for layer in model.layers:
                model.trainable=False
        elif (freeze_till is not None) and freeze_till>0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable=False
        
        flatten_in=tf.keras.layers.Flatten()(model.output)
        prediction=tf.keras.layers.Dense(
            activation="softmax",
            units=classes
        )(flatten_in)

        full_model=tf.keras.models.Model(inputs=model.input,outputs=prediction)

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model=self._prepare_base_model(
            free_all=True,
            freeze_till=None,
            model=self.model,
            learning_rate=self.config.params_learning_rate,
            classes=self.config.params_classes
        )
        self.save_model(path=self.config.updated_base_model_path,model=self.full_model)