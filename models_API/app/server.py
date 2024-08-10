from fastapi import FastAPI
import joblib
import numpy as np
from utils import SwisstopoTileFetcher, ConvertRGBToFeedModel

model = joblib.load('app/HyperUnet_retrain_augmented_noise_corrected_Adam.joblib')

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message': "Colorization Model API"}

@app.post('/predict')
def predict(data: dict):
    # 1) convert image to lab
    # 2) extract lab channel
    # 3) pass lab channel to model
    # use model to make prediction
    prediction = model.predict(data)
    # 4) convert prediction form lab to RGB
    # 5) return image
    return {'inference_image': 'my output'}