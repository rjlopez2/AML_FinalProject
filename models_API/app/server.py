from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from app.utils import SwisstopoTileFetcher, ConvertRGBToFeedModel, create_grayscale_tiles
from toloboy.toloboy import from_LAB_to_RGB_img
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Model paths
model_paths = {
    "HyperUnet_retrain_augmented_noise_corrected_Adam": "app/HyperUnet_retrain_augmented_noise_corrected_Adam.joblib",
    "HyperUnet_retrain_no_augmented_corrected_Adam": "app/HyperUnet_retrain_no_augmented_corrected_Adam.joblib",
    "base_model": "app/base_model.joblib"
}

# Load models into a dictionary
models = {name: joblib.load(path) for name, path in model_paths.items()}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the main page with the form for user input.
    """
    return templates.TemplateResponse("index.html", {"request": request, "model_names": model_paths.keys()})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    coordinates: str = Form(...),
    zoom_level: int = Form(...),
    layer_name: str = Form(default="ch.swisstopo.swissimage"),
    times: str = Form(default="current"),
    model_name: str = Form(...),
):
    """
    Handles the prediction request from the web form, fetches the tiles,
    processes them, and returns the original and colorized images.
    """

    if model_name not in models:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, 
             "error": "Invalid model name", 
             "model_names": model_paths.keys()
             })

    model = models[model_name]

    # Convert form data into required formats
    coordinates = [tuple(map(float, coord.split(','))) for coord in coordinates.split(';')]
    times = times.split(',')

    # Fetch the tiles
    tile_fetcher = SwisstopoTileFetcher(coordinates, zoom_level, layer_name, times)
    tile_fetcher.fetch_all_tiles()
    fetched_tiles = tile_fetcher.get_tiles()

    # Convert fetched tiles to grayscale and prepare for the model
    grayscale_tiles = create_grayscale_tiles(fetched_tiles)

    predictions = {}

    for (longitude, latitude), image_list in grayscale_tiles.items():
        predictions[(longitude, latitude)] = []
        for time, image, image_color, orig_L, origi_AB in image_list:
            # Prepare the image for model input
            # converter = ConvertRGBToFeedModel()
            print(f"Image type: {image}") 
            # print(f"Image shape: {image.shape}") 
            # L, _ = converter.forward(image)
            
            # L= np.asarray(image)
            # print(f"L shape: {L.shape}") 

            # Run the model prediction
            predicted_AB = model.predict(orig_L.reshape(1, *orig_L.shape), verbose=0)
            # print(f"Predicted predicted_AB shape: {predicted_AB.shape}") 
            # print(f"Predicted predicted_AB type: {type(predicted_AB)}") 
            # print(f"Sample predicted_AB type: {predicted_AB}") 

            # Convert predictions back to image format
            # predicted_image = (prediction[0] * 255).astype(np.uint8)
            predicted_image = from_LAB_to_RGB_img(L=orig_L, AB=predicted_AB)
            # print(f"Predicted image shape: {predicted_image.shape}") 
            # print(f"Predicted Image type: {type(predicted_image)}") 
            # print(f"sample predicte Image: {predicted_image[:5, :5, ...]}") 
            predicted_image_pil = Image.fromarray(predicted_image, mode="RGB")

            # Convert images to base64
            original_image_base64 = image_to_base64(image_color)
            original_image_grey_base64 = image_to_base64(image)
            predicted_image_base64 = image_to_base64(predicted_image_pil)
            # predicted_image_base64 = predicted_image_pil

            # Store the results
            predictions[(longitude, latitude)].append({
                "time": time,
                "original_color_image": original_image_base64,
                "original_grey_image": original_image_grey_base64,
                "predicted_image": predicted_image_base64,
            })

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "predictions": predictions, 
        "model_names": model_paths.keys() # Ensure this is always passed
        })


def image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL image to a base64 encoded string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
