# This will make the api backend (fastapi) that receives post requests (images) and returns the prediction of keras tensorflow model
from typing import Annotated
import io

# from model import MyModel

from fastapi import FastAPI, File
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.data_augmentation_1 = kwargs.pop('data_augmentation_1', None) or layers.RandomFlip("horizontal")
        self.data_augmentation_2 = kwargs.pop('data_augmentation_2', None) or layers.RandomRotation(factor=0.1)

        self.inp_norm = layers.Normalization()

        self.conv = kwargs.pop("conv", None) or keras.applications.mobilenet.MobileNet(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            #pooling="avg"
            )

        self.conv.trainable = False

        self.flatten = layers.Flatten()
        self.dense_1 = kwargs.pop("dense_1", None) or layers.Dense(128, activation=keras.activations.mish)
        self.dense_2 = kwargs.pop("dense_2", None) or layers.Dense(128, activation=keras.activations.mish)
        self.dense_3 = kwargs.pop("dense_3", None) or layers.Dense(128, activation=keras.activations.mish)
        self.dense_4 = kwargs.pop("dense_4", None) or layers.Dense(128, activation=keras.activations.mish)

        self.batch_norm_1 = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()
        self.batch_norm_3 = layers.BatchNormalization()
        self.batch_norm_4 = layers.BatchNormalization()
        self.batch_norm_5 = layers.BatchNormalization()

        self.output_layer = kwargs.pop("output_layer", None) or layers.Dense(5)

    def call(self, inputs, training=False):
        x = inputs

        if training:
            x = self.data_augmentation_1(x, training=training)
            x = self.data_augmentation_2(x, training=training)

        x = self.inp_norm(x)

        x = self.conv(x)
        x = self.batch_norm_1(x)
        x = self.flatten(x)

        x = self.dense_1(x)
        x = self.batch_norm_2(x)
        x = self.dense_2(x)
        x = self.batch_norm_3(x)
        x = self.dense_3(x)
        x = self.batch_norm_4(x)
        x = self.dense_4(x)
        x = self.batch_norm_5(x)

        x = self.output_layer(x)

        return x

app = FastAPI()

loaded_model = MyModel()
loaded_model.predict(np.zeros((1, 224, 224, 3)))
loaded_model.load_weights("mystroke_moblienet_87_224_weights.h5")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(image: Annotated[bytes, File()]):
    img = Image.open(io.BytesIO(image))
    img_array = np.array(img)
    img_array = tf.image.resize(img_array, (224, 224))
    prediction = loaded_model.predict(np.expand_dims(img_array, axis=0))
    prediction = layers.Softmax()(prediction)
    prediction = np.array(prediction[0])
    return {"prediction": prediction.tolist()}
