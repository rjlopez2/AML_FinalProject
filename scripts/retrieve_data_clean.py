# %%
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from matplotlib import pyplot as plt
from glob import glob
import os
from copy import deepcopy

import numpy as np
import cv2
from PIL import Image
from time import sleep

import torch
from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import transforms

# Lorenz's libs
import math
import pandas as pd
import requests
from io import BytesIO
from pyproj import Proj, Transformer
import random
from tqdm import tqdm
import folium
from folium.plugins import MarkerCluster

import concurrent.futures

import cProfile

# %% [markdown]
# # Define helper functions/classes

# %% [markdown]
# **SwisstopoTileFetcher Class**
# 
# This class facilitates fetching map tiles from the Swisstopo WMTS service. It converts geographic coordinates (latitude and longitude) into tile indices, constructs the appropriate URL for the tile image, and downloads the image. The class also provides a method to display the fetched tile image using matplotlib.
# 
# Key Methods:
# 
# *   **lat_lon_to_tile_indices():** Converts latitude and longitude to tile indices based on the zoom level.
# *   **fetch_tile():** Downloads the tile image from Swisstopo.
# *   **show_tile():** Displays the fetched tile image.
# 
# Parameters:
# 
# 
# 
# *   **longitude:** The longitude of the point for which the tile is to be fetched.
# *   **latitude:** The latitude of the point for which the tile is to be fetched.
# *   **zoom_level:** The zoom level for the map tile.

# %%
class SwisstopoTileFetcher:
    def __init__(self, longitude, latitude, zoom_level):
        self.scheme = "https"
        self.server_name = "wmts0.geo.admin.ch"  # Can be wmts0 to wmts9
        self.version = "1.0.0"
        self.layer_name = "ch.swisstopo.swissimage"
        self.style_name = "default"
        self.time = "current"
        self.tile_matrix_set = "3857"
        self.format_extension = "jpeg"
        self.longitude = longitude
        self.latitude = latitude
        self.zoom_level = zoom_level

    def lat_lon_to_tile_indices(self):
        n = 2 ** self.zoom_level
        lat_rad = math.radians(self.latitude)
        x_tile = int((self.longitude + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return x_tile, y_tile

    def fetch_tile(self):
        # Convert coordinates to tile indices
        x, y = self.lat_lon_to_tile_indices()

        # Construct the URL
        url = f"{self.scheme}://{self.server_name}/{self.version}/{self.layer_name}/{self.style_name}/{self.time}/{self.tile_matrix_set}/{self.zoom_level}/{x}/{y}.{self.format_extension}"

        # Download the tile

        with requests.Session() as session:
            with session.get(url) as response:
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    return image, url
                else:
                    print(f"Failed to download tile. Status code: {response.status_code}")
                    return None

    def show_tile(self):
        image = self.fetch_tile()
        if image:
            # Display the image
            plt.imshow(image)
            plt.axis('off')  # Hide the axis
            plt.show()

# %% [markdown]
# **ArealstatistikSampler Class**
# 
# This class is designed to sample geographic points from a dataset provided in LV95 coordinates and convert them to WGS84 coordinates. It reads the CSV file*, filters the data based on a specified column, and randomly selects a given number of points from each unique value in that column. The selected points are then transformed from the LV95 coordinate system to the WGS84 coordinate system.
# 
# Key Methods:
# 
# 
# *   **lv95_to_wgs84(lon, lat):** Converts coordinates from LV95 to WGS84.
# *   **sample_points():** Samples the specified number of points for each unique value in the specified column, converts their coordinates, and returns a list of these points.
# 
# 
# Parameters:
# 
# 
# 
# *   **file_path:** Path to the CSV file containing the data.
# *   **column_to_filter:** Column name used to filter and categorize the data.
# *   **num_samples:** Number of samples to select for each unique value in the column.
# *   **random_state:** Optional parameter to ensure reproducibility of random sampling.
# 
# *available on https://www.bfs.admin.ch/bfs/en/home/services/geostat/swiss-federal-statistics-geodata/land-use-cover-suitability/swiss-land-use-statistics.html

# %%
class ArealstatistikSampler:
    def __init__(self, file_path, column_to_filter, num_samples, random_state=None):
        self.file_path = file_path
        self.column_to_filter = column_to_filter
        self.num_samples = num_samples
        self.random_state = random_state

    def lv95_to_wgs84(self, lon, lat):
        in_proj = Proj("epsg:2056")
        out_proj = Proj("epsg:4326")
        transformer = Transformer.from_proj(in_proj, out_proj)
        lon_wgs84, lat_wgs84 = transformer.transform(lon, lat)
        return lon_wgs84, lat_wgs84

    def sample_points(self):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.file_path, delimiter=";")

        # Filter out rows with missing values in the specified column
        df_filtered = df.dropna(subset=[self.column_to_filter])

        # Create an empty list to store the selected points
        selected_points = []

        # Set random state if provided
        if self.random_state is not None:
            random_state = self.random_state
        else:
            random_state = 42  # Default random state
            
        n_classes = df_filtered[self.column_to_filter].unique()

        # Iterate over each unique value in the specified column
        for class_value in n_classes:
            # Filter rows for the current class value
            class_df = df_filtered[df_filtered[self.column_to_filter] == class_value]

            # Randomly select specified number of examples for the current class value
            selected_samples = class_df.sample(n=self.num_samples, random_state=random_state)

            # Convert LV95 coordinates to WGS84 and store them in the selected_points list
            for _, row in selected_samples.iterrows():
                lon_wgs84, lat_wgs84 = self.lv95_to_wgs84(row["E_COORD"], row["N_COORD"])
                selected_points.append([lon_wgs84, lat_wgs84, class_value])

        return selected_points


# %% [markdown]
# Define Parameters

# %%
# file_path = "/content/drive/MyDrive/CAS Avanced Machine Learning/Luftbild_Colarization/ag-b-00.03-37-area-csv.csv"
file_path = "/Volumes/Ruben/datasets/land_use_data/ag-b-00.03-37-area-csv.csv"
# column_to_filter = "AS18_72" #column in the dataset with the classes
column_to_filter = "AS18_17" #column in the dataset with the classes (less classe : 17)
num_samples = 180 #number of samples per class, NOTE: I found out that the lower number of samples from class is 461, so larger number than this will give an error!
random_state = 42
zoom_levels = [16, 17, 18] #zoom levels to fetch images from randomly

# %% [markdown]
# Collect sample points and show the spatial distribution on a map

# %%
# Instantiate ArealstatistikSampler and sample points
sampler = ArealstatistikSampler(file_path, column_to_filter, num_samples, random_state)
coordinates = sampler.sample_points()

# Print the number of samples collected
print("Number of samples collected:", len(coordinates))

# %% [markdown]
# Try to make a faster fetcher class with concurrence

# %%
def fetch_images_with_random_zoom_levels_faster(point, indx, zoom_levels, save_to = None):
    """
    Fetch images for sampled points using random zoom levels.

    Args:
        sampled_points (list): List of sampled points, where each point is represented as a list [lat, lon] or [lat, lon, class_value].
        zoom_levels (list): List of zoom levels to choose from.
        save_to (str): if provided, valid path where to save the  fetched image.

    Returns:
        list: List of dictionaries, each containing fetched image and its metadata (lat, lon, zoom_level, class).
    """

    lat, lon = point[:2]  # Extract latitude and longitude
    class_value = point[2] if len(point) > 2 else None
    zoom_level = random.choice(zoom_levels)
    tile_fetcher = SwisstopoTileFetcher(lon, lat, zoom_level)
    image, url = tile_fetcher.fetch_tile()
    sleep(0.15)
    image_data = {
        'img_id': indx,
        'img_name': '_'.join(url.split("/")[-4:]).split(".")[0],
        'image': image,
        'latitude': lat,
        'longitude': lon,
        'zoom_level': zoom_level,
        'class': class_value,
        'link':url, 
    }

    if save_to:
        assert isinstance(save_to, str), "Path must be a valid string."
        assert os.path.exists(save_to), f"The path proveided '{save_to}' was not found. Make sure that there exists!"

        data_path = os.path.join(save_to, "data")
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
        # print(f"saving img_id: {indx}")
        image.save(os.path.join(data_path, f"img_id_{image_data['img_id']}.jpg"))
           

    return image_data



# %% [markdown]
# Here we set multithreading to fetch images concurrently.
# This allows to speed significanlthly the download time of the images, when is a large number ~10.000 or so.

# %%

# import random
# import os

def fetch_images_concurrently(sampled_points, zoom_levels, save_to=None):
    """
    Fetch images for sampled points concurrently using random zoom levels.

    Args:
        sampled_points (list): List of sampled points, where each point is represented as a list [lat, lon] or [lat, lon, class_value].
        zoom_levels (list): List of zoom levels to choose from.
        save_to (str): Optional path where to save the fetched images.

    Returns:
        list: List of dictionaries, each containing fetched image and its metadata (lat, lon, zoom_level, class).
    """
    # def worker(point, indx, zoom_level):
    #     return fetch_images_with_random_zoom_levels_faster(point, indx, zoom_level, save_to)
    def worker(point, indx):
        return fetch_images_with_random_zoom_levels_faster(point, indx, zoom_levels, save_to)

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_point = {executor.submit(worker, point, indx): (point, indx) for indx, point in enumerate(sampled_points)}
        # future_to_point = {executor.submit(worker, point, indx, zoom_level): (point, indx, zoom_level) for indx, (point, zoom_level) in enumerate(zip(sampled_points, zoom_levels))}
        for future in tqdm(concurrent.futures.as_completed(future_to_point), total=len(future_to_point), desc="Fetching images"):
            point, indx = future_to_point[future]
            # point, indx, zoom_level = future_to_point[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error fetching image for point {point} with index {indx}: {e}")
    
    if save_to:
        print("saving metadata")
        my_df = pd.DataFrame([ {key: d[key] for key in d if key != "image"} for d in results])
        my_df.sort_values(by=['img_id'], ignore_index=True, inplace=True)
        my_df.to_csv(os.path.join(save_to, "metadata.csv"))

    print("#### Done ####")
    return results


# %% [markdown]
# Fetch the images from the given coordinates in the dataset

# %%
# cProfile.run('fetch_images_concurrently(coordinates, zoom_levels, r"/Volumes/Ruben/datasets/fetched_raw_imgs_via_api_full")')

# %%
# Fetch the images
# Fetch the images
path_to_save_raw_images = r"/Volumes/Ruben/datasets/fetched_raw_imgs_via_api_full"
# path_to_save_raw_images = r"/Volumes/Ruben/datasets/fetched_raw_imgs_via_api"
    
# fetched_images = fetch_images_with_random_zoom_levels(coordinates, zoom_levels, save_to=path_to_save_raw_images)
fetched_images = fetch_images_concurrently(coordinates, zoom_levels, path_to_save_raw_images)
len(fetched_images)

# %% [markdown]
# ## debugging problematic images

# %% [markdown]
# The following code snippets are for debuggin purpose to investigate when retrived images were corrupted/missed.
# it is experimentl, so do not run the next code chunks.

# %%
path_to_data = os.path.join(path_to_save_raw_images, "data")
os.path.exists(path_to_data)

# %%
len(coordinates)

# %%
coordinates[0:5]

# %%
df = pd.read_csv(os.path.join(path_to_save_raw_images, "metadata.csv"), index_col=0)
df.head(10)

# %%
df['img_id'].duplicated()

# %%
df[df['img_id'].duplicated()]

# %%
df[df['img_name'].duplicated()]

# %%
df[df['link'].duplicated()]

# %%
img_with_issues_id = [14, 188]
df[df['img_id'].isin(img_with_issues_id)]

# %%
img_with_issues_id = df[df['img_id'].isin(img_with_issues_id)]["img_name"].values
img_with_issues_id

# %%
df[df['img_name'].isin(img_with_issues_id)]

# %%
new_coordinates = [[df.iloc[i]["latitude"], df.iloc[i]["longitude"], df.iloc[i]["class"]]  for i, rows in df[df['img_name'].isin(img_with_issues_id)].iterrows()]
new_coordinates

# %%
df.iloc[0]["latitude"]

# %%
np.array(new_coordinates).shape

# %%
new_zoom_levels = df[df['img_name'].isin(img_with_issues_id)]["zoom_level"].values
new_zoom_levels

# %%
# fetched_images = fetch_images_concurrently(new_coordinates, 
#                                         #    new_zoom_levels,
#                                        #  zoom_levels, 
#                                         [18], 
#                                            os.path.join(path_to_save_raw_images, "test"))
fetched_images = fetch_images_with_random_zoom_levels(new_coordinates, 
                                        #    new_zoom_levels,
                                       #  zoom_levels, 
                                        [18], 
                                           os.path.join(path_to_save_raw_images, "test"))
len(fetched_images)

# %%
zoom_levels

# %%
path_to_save_raw_images

# %%
import pickle

# %%
with open("HyperUnet_retrain_augmented_noise_corrected_Adam_history_dict", 'rb') as my_file:
    history = pickle.load(my_file)

# %%
history

# %%



