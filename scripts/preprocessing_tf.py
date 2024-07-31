# %% [markdown]
# # Load libraries

# %%
# %config InlineBackend.figure_format = 'retina'

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [5, 5]
from glob import glob
import os
from copy import deepcopy
import pickle

import numpy as np
from skimage import util
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.transforms import v2, functional
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.optim import Adam, rmsprop
from warnings import warn


import tensorflow as tf
from tensorflow import keras
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io


from urllib.request import urlretrieve

# Lorenz's libs
# import math
import pandas as pd
import requests
from io import BytesIO
# from pyproj import Proj, Transformer
import random
# from tqdm import tqdm
# import folium
# from folium.plugins import MarkerCluster

from toloboy.toloboy import RGB2LAB

# %%
print(tf.__version__)

# %% [markdown]
# # Define helper functions/classes

# %%

class LTransformation(object):
    def __init__(self, contrast_range=(0.9, 1), brightness_range=(-0.05, 0.20), noise_var_range=(0, 0.005)):
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.noise_var_range = noise_var_range

    def _apply_factor(self, L_channel, contrast_factor, brightness_factor):
        # Apply adjusted brightness and contrast to the L channel
        L_adjusted = contrast_factor * L_channel + brightness_factor

        # Clip adjusted L channel to [0, 1]
        L_adjusted = np.clip(L_adjusted, 0, 1)

        return L_adjusted

    def _apply_noise(self, L_channel, noise_var):
        # Apply Gaussian noise to the L channel
        L_noisy = util.random_noise(L_channel, mode='gaussian', var=noise_var)

        # Clip noisy L channel to [0, 1]
        L_noisy = np.clip(L_noisy, 0, 1)

        return L_noisy

    def _randomize_factors(self):
        return np.random.uniform(*self.brightness_range), np.random.uniform(*self.contrast_range)

    def _randomize_noise_var(self):
        return np.random.uniform(*self.noise_var_range)

    def __call__(self, L_channel):
        while True:
            brightness_factor, contrast_factor = self._randomize_factors()
            noise_var = self._randomize_noise_var()

            # Apply adjusted brightness and contrast to the L channel
            L_adjusted = self._apply_factor(L_channel, contrast_factor, brightness_factor)

            # Apply Gaussian noise to the L channel
            L_augmented = self._apply_noise(L_adjusted, noise_var)

            # Check if values are within range
            if 0 <= np.min(L_augmented) <= np.max(L_augmented) <= 1:
                break

        return L_augmented, contrast_factor, brightness_factor, noise_var

def convert_RGB_to_feed_model(img):
    img = np.asarray(img)
    sz_x = img.shape[0]
    sz_y = img.shape[1]

    train_imgs = np.zeros((sz_x, sz_y, 2))
    train_input = np.zeros((sz_x, sz_y, 1))

    R1 = np.reshape(img[:, :, 0], (sz_x * sz_y, 1))
    G1 = np.reshape(img[:, :, 1], (sz_x * sz_y, 1))
    B1 = np.reshape(img[:, :, 2], (sz_x * sz_y, 1))
    L, A, B = RGB2LAB(R1, G1, B1)

    train_input[:, :, 0] = L.reshape((sz_x, sz_y))
    train_imgs[:, :, 0] = np.reshape(A, (sz_x, sz_y))
    train_imgs[:, :, 1] = np.reshape(B, (sz_x, sz_y))

    return train_input, train_imgs


def convert_RGB__and_augment_to_feed_model(img):
    img = np.asarray(img)
    sz_x = img.shape[0]
    sz_y = img.shape[1]

    train_imgs = np.zeros((sz_x, sz_y, 2))
    train_input = np.zeros((sz_x, sz_y, 1))

    R1 = np.reshape(img[:, :, 0], (sz_x * sz_y, 1))
    G1 = np.reshape(img[:, :, 1], (sz_x * sz_y, 1))
    B1 = np.reshape(img[:, :, 2], (sz_x * sz_y, 1))
    L, A, B = RGB2LAB(R1, G1, B1)

    # Apply LTransformation to the L channel
    L_transformation = LTransformation()
    L_augmented, _, _, _ = L_transformation(L.reshape((sz_x, sz_y)))

    train_input[:, :, 0] = L_augmented
    train_imgs[:, :, 0] = np.reshape(A, (sz_x, sz_y))
    train_imgs[:, :, 1] = np.reshape(B, (sz_x, sz_y))

    return train_input, train_imgs

# %% [markdown]
# # Define custom Dataset class
# 

# %%
class SwisstopoDataset:
    def __init__(self, img_indx, transform=None, large_dataset=False, return_label=True, batch_size=32, shuffle=False):
        self.img_indx = img_indx
        self.transform = transform
        self.large_dataset = large_dataset
        self.return_label = return_label
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Set the appropriate port based on the dataset size
        self.port = 1986 if self.large_dataset else 1985

        # Load metadata
        self.metadata_file = self._load_metadata()

    def _load_metadata(self):
        raw_data_csv_file_link = f"https://perritos.myasustor.com:{self.port}/metadata.csv"
        return pd.read_csv(raw_data_csv_file_link, index_col=0)

    def _fetch_image(self, img_id):
        img_in_server_link = f"https://perritos.myasustor.com:{self.port}/data/img_id_{img_id}.jpg"
        response = requests.get(img_in_server_link)
        image = Image.open(BytesIO(response.content))
        return image

    def _process_image(self, img_id):
        try:
            image = self._fetch_image(img_id)
        except Exception as e:
            warn(f"{'*'*5} Problem loading image id: {img_id} {'*'*5}")
            raise e
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0  # Default normalization
        return image

    def _get_label(self, idx):
        return self.metadata_file["class"].iloc[idx]

    def _generator(self):
        if self.shuffle:
            img_indices = np.random.permutation(len(self.img_indx))
        else:
            img_indices = self.img_indx

        for idx in range(len(self.img_indx)):
            image = self._process_image(self.img_indx[idx])
            L, AB = image  # Unpack the transformed image
            if self.return_label:
                label = self._get_label(idx)
                yield (L, AB), label
            else:
                yield L, AB

    def get_dataset(self):
        # Dynamically infer the shapes of L and AB channels
        def _dynamic_output_signature():
            example_image = self._fetch_image(self.img_indx[0])
            example_transformed = self.transform(example_image)
            L, AB = example_transformed
            L_shape = tf.TensorSpec(shape=L.shape, dtype=tf.float32)
            AB_shape = tf.TensorSpec(shape=AB.shape, dtype=tf.float32)
            if self.return_label:
                return ((L_shape, AB_shape), tf.TensorSpec(shape=(), dtype=tf.int64))
            else:
                return (L_shape, AB_shape)

        output_signature = _dynamic_output_signature()

        dataset = tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
        dataset = dataset.batch(self.batch_size, drop_remainder=True) # use drop reminder to have same size always
        return dataset

# %% [markdown]
# # Define transforms
# 

# %%
def convert_to_LAB_transform(image):
    L, AB = convert_RGB_to_feed_model(image)
    return (L, AB)

# %%
def convert_to_LAB_and_augment_transform(image):
    if np.random.rand() < 0.25:  # only apply augmentation to 25% of the data
        L, AB = convert_RGB__and_augment_to_feed_model(image)
    else:
        L, AB = convert_RGB_to_feed_model(image)
    return (L, AB)

# %% [markdown]
# # Check info from the images

# %% [markdown]
# The data was initially created using the scripts `retrieve_data.ipynb` and stored in a private server for later (re)use.
# In the metadata.csv file we get the information on original link, class and coordinates of each image.
# 
# NOTE: the following are links stored in a private server, jet they are still publically available.

# %%
is_large_dataset = True

if is_large_dataset:
    server_port = 1986 # Large dataset of ~10K images
else:
    server_port = 1985 # Large dataset of ~10K images
# server_port = 1985 # initial dataset of 3.6K images

raw_data_csv_file_link = f"https://perritos.myasustor.com:{server_port}/metadata.csv"


metadata_raw_df = pd.read_csv(raw_data_csv_file_link, index_col=0)
metadata_raw_df.info()

# %% [markdown]
# # Split the Train, Valid and Test subsets.

# %% [markdown]
# We use the column `image_id` from the metadata as index of the images and then we perform standard shufling and splitting.
# 
# ~~The final ratio for the train, validation and test dastasets are: 70, 29 and 1 % respectively~~
# 
# The final ratio for the train, validation and test dastasets are: 59, 25 and 16 % respectively.
# 
# NOTE: we add additonal ~3000 images for computing the FID score metrics, rusulting in roughly using 10K images for training/valiation (in a ratio of 70/30% repectively) and 3K images for testing. Total images used were 12960.
# 

# %%
dataX, dataY = metadata_raw_df["img_id"].to_list(), metadata_raw_df["class"] .to_list()

rand_state = 9898

train_ratio = 0.5866
validation_ratio = 0.2514
test_ratio = 0.162



# train is now 75% of the entire data set
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, stratify = dataY, random_state=rand_state)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), stratify=y_test, random_state=rand_state)

print(f"the size fo the train dataset is: {len(x_train)}.\nthe size fo the validation dataset is: {len(x_val)}.\nthe size fo the test dataset is: {len(x_test)}.")

# %%
b_size =64

# Instantiate the dataset
# img_indices = [0, 1, 2, 3, 4, 5]  # Example indices

train_dataset_loader = SwisstopoDataset(x_train,
                           transform=convert_to_LAB_and_augment_transform,
                           large_dataset=True,
                           return_label=False,
                           batch_size=b_size,
                           shuffle=True)

valid_dataset_loader = SwisstopoDataset(x_val,
                           transform=convert_to_LAB_transform,
                           large_dataset=True,
                           return_label=False,
                           batch_size=b_size,
                           shuffle=False)

test_dataset_loader = SwisstopoDataset(x_test,
                           transform=convert_to_LAB_transform,
                           large_dataset=True,
                           return_label=False,
                           batch_size=b_size,
                           shuffle=False)

train_dataset = train_dataset_loader.get_dataset()
test_dataset = test_dataset_loader.get_dataset()
valid_dataset = valid_dataset_loader.get_dataset()


# %%
# Get the tf.data.Dataset
# Iterate over the dataset
for batch in train_dataset:
    # (L_channel, AB_channels), labels = batch # print with labels
    # print(L_channel.shape, AB_channels.shape, print(labels.shape))
    L_channel, AB_channels= batch # print without labels
    print(L_channel.shape, AB_channels.shape)
    break


