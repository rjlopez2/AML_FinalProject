# %% [markdown]
# # Load libraries

# %%
# !pip install --upgrade torch
# !pip install --upgrade tensorflow
# !pip install --upgrade jax
# !pip install --upgrade keras-nlp
# !pip install --upgrade keras-cv
# !pip install --upgrade keras

# %%
# %config InlineBackend.figure_format = 'retina'

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [5, 5]
import os
from urllib import request
import numpy as np
from skimage import util
from PIL import Image
import math
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import requests
from io import BytesIO
from toloboy.toloboy import RGB2LAB, from_LAB_to_RGB_img
from tqdm import tqdm
tf.experimental.numpy.experimental_enable_numpy_behavior()
from skimage.color import deltaE_ciede2000
import cv2

from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

# %%
print(tf.__version__)

# %% [markdown]
# # Define helper functions/classes

# %% [markdown]
# ## Metrics functions

# %%

# def mse(imageA, imageB, nband = 3):
# 	# the 'Mean Squared Error' between the two images is the
# 	# sum of the squared difference between the two images;
# 	# NOTE: the two images must have the same dimension
# 	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
# 	err /= float(imageA.shape[0] * imageA.shape[1] * nband)
	
# 	# return the MSE, the lower the error, the more "similar"
# 	# the two images are
# 	return err


# %%
# def rmse(imageA, imageB, nband):
# 	# the 'Root Mean Squared Error' between the two images is the
# 	# sum of the squared difference between the two images;
# 	# NOTE: the two images must have the same dimension
# 	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
# 	err /= float(imageA.shape[0] * imageA.shape[1] * nband)
# 	err = np.sqrt(err)
# 	return err

# %%

def mae(imageA, imageB, bands = 3):
	# the 'Mean Absolute Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
	return err
        

# %%

def psnr(img1, img2):
    # Compute The peak signal-to-noise ratio (PSNR)
    # Higher PSNR values indicate a higher quality of the predicted image.
    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)
    mse = np.mean( (img1 - img2) ** 2 )
    # print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# %%
def compute_delta_e_cie2000(img0_rgb,imag1_RGB,Kl=1, KC=1, KH=1):
    
    Lab1 = cv2.cvtColor(img0_rgb, cv2.COLOR_BGR2Lab)
    Lab2 = cv2.cvtColor(imag1_RGB, cv2.COLOR_BGR2Lab)
    L1, a1, b1 = cv2.split(Lab1)
    L2, a2, b2 = cv2.split(Lab2)
    
    
    delta=deltaE_ciede2000(L1,L2, Kl, KC, KH)
    #print(len(delta))
    
    return np.mean(delta)

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
b_size = 1

# Instantiate the dataset
# img_indices = [0, 1, 2, 3, 4, 5]  # Example indices



test_dataset_loader = SwisstopoDataset(x_test,
                           transform=convert_to_LAB_transform,
                           large_dataset=True,
                           return_label=True,
                           batch_size=b_size,
                           shuffle=False)

# train_dataset = train_dataset_loader.get_dataset()
test_dataset = test_dataset_loader.get_dataset()
# valid_dataset = valid_dataset_loader.get_dataset()


# %%
# Get the tf.data.Dataset
# Iterate over the dataset
# for batch in test_dataset:
#     # (L_channel, AB_channels), labels = batch # print with labels
#     # print(L_channel.shape, AB_channels.shape, print(labels.shape))
#     L_channel, AB_channels= batch # print without labels
#     print(L_channel.shape, AB_channels.shape)
#     break

# %% [markdown]
# # Load Models

# %% [markdown]
# Additional info about the base *Hyper-U-Net* model can be found at the following sources:
# 
# - link to [original paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9604844/)
# 
# - link to [repository](https://github.com/3DOM-FBK/Hyper_U_Net?tab=readme-ov-file)
# 
# - link to [model]("https://drive.usercontent.google.com/download?id=19DaA9f1HIOW9PmUz11xKw65fCo3X7-Fw&export=download&authuser=0&confirm=t&uuid=8a03b6f8-6f5d-4bc8-a62d-8b0cfc98d2db&at=APZUnTU9WqjmYlQcAGh22O2M8wXI%3A1717452655512")
# 
# NOTE: This will download a multiples large files to your device in the current directory.

# %% [markdown]
# Check if model is in the current directory otherwise download it.

# %%
models_sources = {
    "model_name": [
        "base_model",
        # "HyperUnet_retrained_30e",
        # "HyperUnet_retrain_augmented_30e",
        # "HyperUnet_retrain_augmented_noise_25e",
        "HyperUnet_retrain_augmented_noise_corrected_Adam",
        # "HyperUnet_retrain_augmented_noise_corrected_rmsprop",
        "HyperUnet_retrain_no_augmented_corrected_Adam",
          ],
    "url": [
        "https://drive.usercontent.google.com/download?id=19DaA9f1HIOW9PmUz11xKw65fCo3X7-Fw&export=download&authuser=0&confirm=t&uuid=8a03b6f8-6f5d-4bc8-a62d-8b0cfc98d2db&at=APZUnTU9WqjmYlQcAGh22O2M8wXI%3A1717452655512",
        # "https://perritos.myasustor.com:1986/Models/HyperUnet_retrained_30e/HyperUnet_retrain1.keras",
        # "https://perritos.myasustor.com:1986/Models/HyperUnet_retrain_augmented/HyperUnet_retrain_augmented_epoch30_valloss0.0011.keras",
        # "https://perritos.myasustor.com:1986/Models/HyperUnet_retrain_augmented_noise/HyperUnet_retrain_augmented_noise_ckpt_epoch25_valloss0.0010.keras",
        "https://perritos.myasustor.com:1986/Models/HyperUnet_retrain_augmented_noise_corrected_Adam/HyperUnet_retrain_augmented_noise_corrected_Adam_ckpt_epoch27_valloss0.0234.keras",
        # "https://perritos.myasustor.com:1986/Models/HyperUnet_retrain_augmented_noise_corrected_rmsprop/HyperUnet_retrain_augmented_noise_corrected_rmsprop_ckpt_epoch30_valloss0.0248.keras",
        "https://perritos.myasustor.com:1986/Models/HyperUnet_retrain_no_augmented_corrected_Adam/HyperUnet_retrain_no_augmented_corrected_Adam_ckpt_epoch27_valloss0.0230.keras"
        ],
    "extension": [
        "h5",
        # "keras",
        # "keras",
        # "keras",
        "keras",
        # "keras",
        "keras",
    ]
}

for i in range(len(models_sources[next(iter(models_sources.keys()))])):
    model_name = models_sources["model_name"][i]
    url = models_sources["url"][i]
    print(f"Model Name: {model_name}\nURL: {url}\n")

# %%
for i in range(len(models_sources[next(iter(models_sources.keys()))])):
    file_name = models_sources["model_name"][i] + "." + models_sources["extension"][i]
    if not os.path.exists(os.path.join(os.curdir,file_name)):
        model_url = models_sources['url'][i]
        # !wget -O {file_name}  "$model_url"
        request.urlretrieve(model_url, file_name)
    else:
        print("########### No model to download, everything is in the current directory ##############")
    # !wget -O {models_sources["model_name"][i] + "." + models_sources["extension"][i]} {models_sources["url"][i]}

# %%

models = {}

for i in range(len(models_sources[next(iter(models_sources.keys()))])):
    file_name = models_sources["model_name"][i] + "." + models_sources["extension"][i]
    # print(f"{'*'*5} loading '{models_sources['model_name'][i]}' {'*'*5}")
    print(f"{'*'*5} loading '{file_name}' {'*'*5}")
    models[models_sources["model_name"][i]] = load_model(file_name)
    print(f"{'*'*5} done {'*'*5}")

# %% [markdown]
# # Compute metrics

# %% [markdown]
# Here we create the main loop to compute the metrics.
# 
# - We use the *Test* tadates of about 100 images.
# 
# - First we pass the L channel (grey image) to the model to make the prediction.
# 
# - Then we transform the **original** AND the **predicted** L\*a\*b images to RGB colorspace and plot them to compare results.
# 
# - Finally we compute a number of metrics and summarize the results in a dataframe

# %%
models

# %% [markdown]
# ## define helper function

# %%

# n_imgs_to_test = len(test_dataset)
def evaluate_model_metrics(dataset, model, model_name, n_samples):

    sample = dataset.take(n_samples)
    deltaE_list = []
    MAE_list = []
    PSNR_list = []
    SSIM_list = []

    for indx, data in enumerate(tqdm(sample, desc = f"evaluating model: '{model_name}'", total=n_samples)):

        (L, AB), label = data
        # print(type(L))
        # # L = tf.squeeze(L)
        # print(L.shape)
        # print(type(AB))
        # print(AB.shape)
        # print(type(label))
        # print(label.shape)
        # my_model = models["base_model"]
        predicted_AB = model.predict(L, verbose=0)
        predicted_RGB = from_LAB_to_RGB_img(L[0, ...], predicted_AB)
        original_RGB = from_LAB_to_RGB_img(L[0, ...], AB)
        # print(f"Shape of original RGB is :{original_RGB.shape}")
        # print("*** Computing metrics ***")
        MAE = mae(original_RGB,predicted_RGB,3)
        PSNR= psnr(original_RGB,predicted_RGB)
        SSIM, _ = ssim(original_RGB, predicted_RGB, channel_axis=2, full=True) # NOTE: need to specify axis channels, otherwise complains!


        
        deltaE = compute_delta_e_cie2000(original_RGB, predicted_RGB)
                    
        MAE_list.append(MAE)
        SSIM_list.append(SSIM)
        PSNR_list.append(PSNR)
        deltaE_list.append(deltaE)


    metrics_dict = {
    "Model": model_name,
    "dE2000": deltaE_list,
    "MAE" : MAE_list,
    "PSNR" : PSNR_list,
    "SSIM" : SSIM_list,
    }

    return metrics_dict


# %% [markdown]
# ## start the metrics evaluation loop

# %%
metrics_all_list = []

for indx, model in enumerate(models):
    print(f"{'*'*5} assesing model: '{model}' {'*'*5}")
    # if indx > 0:
    # for sample in tqdm(data_to_test):
    result = evaluate_model_metrics(dataset = test_dataset, model=models[model], model_name=model, n_samples = 100)

    metrics_all_list.append(result)
    print(f"{'*'*5} done {'*'*5}")
        # break
    # break

# %% [markdown]
# ## Wrap and reformat results in a table

# %%
# numeric_columns = ["MSE", "PSNR", "MSEr", "MSEg", "MSEb", "RMSE", "SSIM"]
numeric_columns = ["dE2000", "MAE", "PSNR", "SSIM"]
metrics_df = pd.DataFrame(metrics_all_list).explode(numeric_columns, ignore_index=True)
metrics_df[numeric_columns] = metrics_df[numeric_columns].astype(float)
metrics_df['Model'] = metrics_df['Model'].str.replace('_', ' ')
metrics_df['Model'] = metrics_df['Model'].str.replace('corrected', ' ')
metrics_df['Model'] = metrics_df['Model'].str.replace('retrain', ' ')
metrics_df['Model'] = metrics_df['Model'].str.replace('noise', ' ')
# metrics_df['Model'] = [f"Model{i}" for i in range(8)]
metrics_df.info()

# %%
metrics_df_grouped = metrics_df.groupby("Model", as_index=False)
metrics_df_grouped

# %% [markdown]
# Compute the mean

# %%

# metrics_df.groupby("Model", as_index=False).mean().round(2).to_csv("metrics_df_mean.csv", index=False)
metrics_df_mean = metrics_df_grouped.mean().round(2)
# metrics_df_mean["Model"] = [f"Model{i}" for i in range(7)]
# metrics_df_mean.to_csv("metrics_df_mean.csv", index=False)
metrics_df_mean


# %% [markdown]
# Compute the SD

# %%

metrics_df_std = metrics_df_grouped.std().round(1)
# metrics_df_mean["Model"] = [f"Model{i}" for i in range(7)]
# metrics_df_std.to_csv("metrics_std.csv", index=False)
metrics_df_std


# %%

metrics_df_mean_plus_std = metrics_df_mean.iloc[:, 1:].astype(str) + u"\u00B1" + metrics_df_std.iloc[:, 1:].astype(str)
metrics_df_mean_plus_std.insert(0, "Model", metrics_df_mean.iloc[:, 0])
metrics_df_mean_plus_std


# %% [markdown]
# Save file

# %%
# metrics_df_mean_plus_std.to_csv("metrics_df_mean_plus_std.csv", index=False)

# %% [markdown]
# ## Calculate FID score
# Adapted from this [sorce](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/) to calculate FID

# %%
# example of calculating the frechet inception distance in Keras for cifar10
# import numpy
# from numpy import cov
# from numpy import trace
# from numpy import iscomplexobj
# from numpy import asarray

# from keras.datasets import cifar10

# %%
def predict_images(dataset, model, model_name, n_samples):

    original_imgs_stack = []
    predicted_imgs_stack = []

    # n_samples = 2100
    sample = dataset.take(n_samples)

    for data in tqdm(sample, desc = f"evaluating model: '{model_name}'", total=n_samples):
        (L, AB), label = data
        predicted_AB = model.predict(L, verbose=0)
        predicted_RGB = from_LAB_to_RGB_img(L[0, ...], predicted_AB)
        predicted_imgs_stack.append(predicted_RGB)
        # print(f"type predicted_RGB: '{type(predicted_RGB)}'. size: {predicted_RGB.shape}")
        
        original_RGB = from_LAB_to_RGB_img(L[0, ...], AB)
        original_imgs_stack.append(original_RGB)
        # print(f"type original_RGB: '{type(original_RGB)}'. size: {original_RGB.shape}")
        # print(f"{'*'*5} Done {'*'*5}")
        # break
    
    return (np.array(original_imgs_stack), np.array(predicted_imgs_stack))



# %%
# orignal_imgs, predicted_imgs = predict_images(data_to_test, models["base_model"], "base_model", n_samples=50)

# %%
# orignal_imgs.shape, predicted_imgs.shape

# %%
# np.apply_over_axes(func = mae, a=[orignal_imgs, predict_images], axes=0)
# np.apply_along_axis(psnr, 0, orignal_imgs,  predict_images)

# %%


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.asarray(images_list)


# %%
# teste = scale_images(orignal_imgs, (299,299,3))
# teste.shape

# %%

# calculate frechet inception distance
def calculate_fid(images1, images2, on_model):
	
	model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
	images1 = images1.astype('float32')
	images2 = images2.astype('float32')
	# resize images
	images1 = scale_images(images1, (299,299,3))
	images2 = scale_images(images2, (299,299,3))
	print('Scaling images', images1.shape, images2.shape)
	# pre-process images
	images1 = preprocess_input(images1) # preprare the data to adjusted for the model input. see reference here: https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/preprocess_input
	images2 = preprocess_input(images2)

	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	print('FID: %.3f' % fid)
	fid = {"Model": model_name,
		   "FID": fid}
	return fid


# %%
# calculate_fid(orignal_imgs, predicted_imgs, "lalala")

# %%
FID_metrics_list = []

for indx, model_name in enumerate(models):
    print(f"{'*'*5} computing FID on model: '{model_name}' {'*'*5}")
    # if indx > 0:
    # for sample in tqdm(data_to_test):
    # result = calculate_fid(data_to_test, models[model], model, n_samples)
    orignal_imgs, predicted_imgs = predict_images(test_dataset, models[model_name], model_name, n_samples=2100)
    FID_metrics_list.append(calculate_fid(orignal_imgs, predicted_imgs, model_name))
    
    print(f"{'*'*5} done {'*'*5}")

FID_metrics_list

# %%
FID_df = pd.DataFrame(FID_metrics_list)
FID_df['Model'] = FID_df['Model'].str.replace('_', ' ')
FID_df['Model'] = FID_df['Model'].str.replace('corrected', ' ')
FID_df['Model'] = FID_df['Model'].str.replace('retrain', ' ')
FID_df['Model'] = FID_df['Model'].str.replace('noise', ' ')

FID_df

# %%
metrics_df_mean_plus_std = pd.merge(metrics_df_mean_plus_std, FID_df)

# %%
metrics_df_mean_plus_std.to_csv("metrics_df_mean_plus_std.csv", index=False)

# %%
# result

# %%



