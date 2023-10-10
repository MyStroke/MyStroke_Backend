# This will make the api backend (fastapi) that receives post requests (images) and returns the prediction of keras tensorflow model
from typing import Annotated
import io

from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loaded_model = tf.saved_model.load("hand_efficientnet_91_224")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(image: Annotated[bytes, File()]):
    img = Image.open(io.BytesIO(image))
    img = img.convert("RGB")

    img_array = np.array(img)
    img_array = tf.image.resize(img_array, (224, 224))

    prediction, _ = loaded_model(np.expand_dims(img_array, axis=0))
    prediction = np.array(prediction[0])
    return {
        "prediction": prediction.tolist(),
        "argmax": np.argmax(prediction).tolist(),
        "top_2": np.argsort(prediction)[-2:].tolist(),
    }
