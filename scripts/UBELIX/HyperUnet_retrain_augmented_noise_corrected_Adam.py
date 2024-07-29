# %% [markdown]
# # Load libraries

# %%
# %config InlineBackend.figure_format = 'retina'

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [5, 5]
from glob import glob
import os
from copy import deepcopy

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
        image = self._fetch_image(img_id)
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
# The final ratio for the train, validation and test dastasets are: 70, 29 and 1 % respectively

# %%
dataX, dataY = metadata_raw_df["img_id"].to_list(), metadata_raw_df["class"] .to_list()

rand_state = 9898
train_ratio = 0.70
validation_ratio = 0.29
test_ratio = 0.01



# train is now 75% of the entire data set
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, stratify = dataY, random_state=rand_state)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), stratify=y_test, random_state=rand_state)

print(f"the size fo the train dataset is: {len(x_train)}.\nthe size fo the validation dataset is: {len(x_val)}.\nthe size fo the test dataset is: {len(x_test)}.")

# %%
b_size = 64

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

# %% [markdown]
# # Load "base" TF model

# %% [markdown]
# Load base Hyper-U-Net model from the original source:
# 
# - link to [original paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9604844/)
# 
# - link to [repository](https://github.com/3DOM-FBK/Hyper_U_Net?tab=readme-ov-file)
# 
# - link to [model]("https://drive.usercontent.google.com/download?id=19DaA9f1HIOW9PmUz11xKw65fCo3X7-Fw&export=download&authuser=0&confirm=t&uuid=8a03b6f8-6f5d-4bc8-a62d-8b0cfc98d2db&at=APZUnTU9WqjmYlQcAGh22O2M8wXI%3A1717452655512")
# 
# NOTE: This will download a .h5 file to your device in the current directory.

# %% [markdown]
# # Retrain Hyper UNet (**with** Transformations)
# 
# Here you start the training loop or resume training with the following logics:
# - search for model name "folder"
# 
# if the model does not exist
# - create a new one and retrieve the *vanilla* HyperUnet model
# - create configuration for checkpoints (folders/frequency/names)
# - start trianin loop
# 
# if the model exists but training was not finished:
# - find the filder with the checkpoints 
# - resume training from last checkpoint until reaching the initial no of epochs

# %%
epochs = 30
name = "HyperUnet_retrain_augmented_noise_corrected_Adam"

checkpoint_dir = os.path.abspath(
    os.path.join(
        os.curdir, 
        "..",
        "models",
        name,
        )
        )
# checkpoint_dir = r"/Volumes/Ruben/datasets/fetched_raw_imgs_via_api_full/Models/HyperUnet_retrain_augmented_noise"
model_ckpt_file_name = f"{name}_ckpt_epoch{{epoch:02d}}_valloss{{val_loss:.4f}}.keras"

checkpoint_path = os.path.join(checkpoint_dir,
                               model_ckpt_file_name)

callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            save_freq='epoch',
            save_best_only=False,
            save_weights_only = False,
            verbose=1
            )
            ]


if not os.path.exists(checkpoint_dir):

    print(f"No model '{name}' found, training from scratch. n_epochs = {epochs}")

    print(f"******* making new dir to store model @: '{checkpoint_dir}' *******")

    os.makedirs(checkpoint_dir)

    print("******* Loading Base model *******")

    url = ("https://drive.usercontent.google.com/download?id=19DaA9f1HIOW9PmUz11xKw65fCo3X7-Fw&export=download&authuser=0&confirm=t&uuid=8a03b6f8-6f5d-4bc8-a62d-8b0cfc98d2db&at=APZUnTU9WqjmYlQcAGh22O2M8wXI%3A1717452655512")
    filename="Hyper_U_Net.h5"

    if not os.path.exists(os.path.join(os.curdir, "Hyper_U_Net.h5")):
        path, headers = urlretrieve(url, filename)
    # for name, value in headers.items():
    #     print(name, value)
    # model1 = keras.models.load_model(os.path.join(os.curdir, "Hyper_U_Net.h5"))
    # Load the saved model
    loaded_model = tf.keras.models.load_model("Hyper_U_Net.h5")

    # Find the index of the last encoder layer
    last_encoder_layer_index = loaded_model.layers.index(loaded_model.get_layer('max_pooling2d_4'))

    # Freeze all layers up to the last encoder layer
    for layer in loaded_model.layers[:last_encoder_layer_index + 1]:
        layer.trainable = False

    # Function to count the number of parameters
    def count_params(model, only_trainable=False):
        if only_trainable:
            return np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        else:
            return np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights + model.non_trainable_weights])

    # Get the number of trainable and non-trainable parameters
    trainable_params = count_params(loaded_model, only_trainable=True)
    total_params = count_params(loaded_model)

    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")

    # Display model summary
    loaded_model.summary()

    # Compile the model after freezing the encoder layers
    # loaded_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-5, 
                                                             decay_steps=10,
                                                             decay_rate=0.96)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loaded_model.compile(
        optimizer = opt,
        # Adam(lr = 2e-6), 
        loss = 'mean_absolute_error', 
        metrics = ['RootMeanSquaredError']
        )

    
    print("******* start training loop *******")
    history = loaded_model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=valid_dataset,
                validation_steps=4,
                callbacks=callbacks,
                verbose=1
            )
    print("******* Done with training *******")

else:

    print(f"******* the model '{name}' was already initialize here *******.\n******* Finding last model to resume training *******")
    model_list = sorted(glob(os.path.join(checkpoint_dir, "*.keras")), 
                        key=lambda x: int([segment for segment in x.split('_') if 'epoch' in segment][0][-2:]))
    
    try:
    
        if len(model_list) > 0:
            
            last_model_path = model_list[-2]
            last_epoch_found = int(last_model_path.split('_')[-2][-2:])
            last_epoch_found_str = str(last_model_path.split('_')[-2][-2:]) # same thing as last_model_path bus as string
            print(f"Restoring model '{name}' with last epoch = {last_epoch_found}")
            # Restore the full model
            print(">>>>>>> here is before error: ",checkpoint_dir)
            print(">>>>>>> here is before error: ",model_ckpt_file_name)
            # latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir, model_ckpt_file_name)
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir, checkpoint_path)
            # latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            print(">>>>>>> here passed: `tf.train.latest_checkpoint(checkpoint_dir, model_ckpt_file_name)` ")
            restored_model = tf.keras.models.load_model(latest_checkpoint)
            print(">>>>>>> here passed: `tf.keras.models.load_model(latest_checkpoint)` ")
            
            # new_checkpoint_dir = os.path.join(checkpoint_dir, f"retrained_from_epoch{last_epoch_found_str}")
            
            # print(f"new models will be stored @ the subfolder {new_checkpoint_dir}.")
            
            epochs = epochs - last_epoch_found
            
            
            print(f"Resume training from last epoch. Remaining : {epochs} - {last_epoch_found} = {epochs - last_epoch_found}.")
            # loaded_model = tf.keras.models.load_model(last_model_path)
            history = restored_model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=valid_dataset,
                validation_steps=4,
                callbacks=callbacks,
                verbose=1
            )
            print("******* Done with training *******")
        else:
            print(f"A folder with model name '{name}' was found but not model inside. \nStarting training.")
            print("******* Loading Base model *******")
            url = ("https://drive.usercontent.google.com/download?id=19DaA9f1HIOW9PmUz11xKw65fCo3X7-Fw&export=download&authuser=0&confirm=t&uuid=8a03b6f8-6f5d-4bc8-a62d-8b0cfc98d2db&at=APZUnTU9WqjmYlQcAGh22O2M8wXI%3A1717452655512")
            filename="Hyper_U_Net.h5"

            if not os.path.exists(os.path.join(os.curdir, "Hyper_U_Net.h5")):
                path, headers = urlretrieve(url, filename)
            # for name, value in headers.items():
            #     print(name, value)
            # model1 = keras.models.load_model(os.path.join(os.curdir, "Hyper_U_Net.h5"))
            # Load the saved model
            loaded_model = tf.keras.models.load_model("Hyper_U_Net.h5")

            # Find the index of the last encoder layer
            last_encoder_layer_index = loaded_model.layers.index(loaded_model.get_layer('max_pooling2d_4'))

            # Freeze all layers up to the last encoder layer
            for layer in loaded_model.layers[:last_encoder_layer_index + 1]:
                layer.trainable = False

            # Function to count the number of parameters
            def count_params(model, only_trainable=False):
                if only_trainable:
                    return np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
                else:
                    return np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights + model.non_trainable_weights])

            # Get the number of trainable and non-trainable parameters
            trainable_params = count_params(loaded_model, only_trainable=True)
            total_params = count_params(loaded_model)

            print(f"Trainable parameters: {trainable_params}")
            print(f"Total parameters: {total_params}")

            # Display model summary
            loaded_model.summary()

            # Compile the model after freezing the encoder layers
            # loaded_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-5, 
                                                                    decay_steps=10,
                                                                    decay_rate=0.96)
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            loaded_model.compile(
                optimizer = opt,
                # Adam(lr = 2e-6), 
                loss = 'mean_absolute_error', 
                metrics = ['RootMeanSquaredError']
                )

            
            print("******* start training loop *******")
            history = loaded_model.fit(
                        train_dataset,
                        epochs=epochs,
                        validation_data=valid_dataset,
                        validation_steps=4,
                        callbacks=callbacks,
                        verbose=1
                    )
            print("******* Done with training *******")
        # else:
        #     raise FileNotFoundError(f"A folder with model name '{name}' was found but not model inside. Define what to do here!!!")
                
        # history = loaded_model.fit(
        #     train_dataset,
        #     epochs=epochs,
        #     validation_data=valid_dataset,
        #     validation_steps=4,
        #     callbacks=callbacks,
        #     verbose=1
        # )
    # model_list = sorted(glob(os.path.join(checkpoint_dir, "*.keras")), 
    #                     key=lambda x: int([segment for segment in x.split('_') if 'epoch' in segment][0][-2:]))


    # if len(model_list) > 0:
    #     last_model_path = model_list[-2]
    #     last_epoch_found = int(last_model_path.split('_')[-2][-2:])
    #     print(f"found model with last epoch = {last_epoch_found}")
    #     print(f"Resume training from last epoch. Remaining : {epochs} - {last_epoch_found} = {epochs - last_epoch_found}.")
    #     epochs = epochs - last_epoch_found
        # print(last_model_path)
        # loaded_model.load_weights(last_model_path)
        # latest = tf.train.latest_checkpoint(last_model_path)
        # latest
        # loaded_model.load_weights(latest)

        # loaded_model = tf.keras.models.load_model(last_model_path)
        # history = loaded_model.fit(
        #     train_dataset,
        #     epochs=epochs,
        #     validation_data=valid_dataset,
        #     validation_steps=4,
        #     callbacks=callbacks,
        #     verbose=1
        # )
    # else:
    #     print(f"No model found, training from scratch n_epochs = {epochs}")
        # history = loaded_model.fit(
        #     train_dataset,
        #     epochs=epochs,
        #     validation_data=valid_dataset,
        #     validation_steps=4,
        #     callbacks=callbacks,
        #     verbose=1
        # )

    except Exception as exc:
        print(f'Fail to load model with error ---->>>> {exc} <<<<----')




# %% [markdown]
# ## start training loop

# %% [markdown]
# 

# %% [markdown]
# ### recap from last checkpoint otherwise start a new training

# %%
# checkpoint_dir = r"/Volumes/Ruben/datasets/fetched_raw_imgs_via_api_full/Models/HyperUnet_retrain_augmented_noise"
# os.path.exists(checkpoint_dir)

# %%


# %%
# Evaluate the model on the test dataset
test_loss, test_accuracy = loaded_model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')


# %% [markdown]
# ## Plotting the training evolution

# %%
# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(checkpoint_dir, f'{name}_loss.png'), dpi=100)
# plt.savefig(sys.stdout.buffer)
plt.show()



# %%

# Plot the training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig(os.path.join(checkpoint_dir, f'{name}_accuracy.png'), dpi=100)
# plt.savefig(sys.stdout.buffer)
plt.show()

# %%
print("######### Training is done! yahoo!!!!! #########")


