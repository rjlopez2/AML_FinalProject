import math
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch

class SwisstopoTileFetcher:
    def __init__(self, coordinates, zoom_level, layer_name="ch.swisstopo.swissimage", times=["current"]):
        self.scheme = "https"
        self.server_name = "wmts.geo.admin.ch"
        self.version = "1.0.0"
        self.layer_name = layer_name
        self.style_name = "default"
        self.tile_matrix_set = "3857"
        self.format_extension = "jpeg"
        self.coordinates = coordinates
        self.zoom_level = zoom_level
        self.times = times
        self.fetched_tiles = {}

    def lat_lon_to_tile_indices(self, longitude, latitude):
        n = 2 ** self.zoom_level
        lat_rad = math.radians(latitude)
        x_tile = int((longitude + 180.0) / 360.0 * n)
        y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return x_tile, y_tile

    def fetch_tile(self, longitude, latitude, time):
        x, y = self.lat_lon_to_tile_indices(longitude, latitude)
        url = f"{self.scheme}://{self.server_name}/{self.version}/{self.layer_name}/{self.style_name}/{time}/{self.tile_matrix_set}/{self.zoom_level}/{x}/{y}.{self.format_extension}"
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            print(f"Failed to download tile. Status code: {response.status_code}")
            return None

    def fetch_all_tiles(self):
        for longitude, latitude in self.coordinates:
            self.fetched_tiles[(longitude, latitude)] = []
            for time in self.times:
                image = self.fetch_tile(longitude, latitude, time)
                if image:
                    self.fetched_tiles[(longitude, latitude)].append((time, image))

    def show_tiles(self):
        self.fetch_all_tiles()
        for (longitude, latitude), image_list in self.fetched_tiles.items():
            fig, axes = plt.subplots(1, len(image_list), figsize=(15, 5))
            fig.suptitle(f"Coordinates: ({longitude, latitude})")
            if len(image_list) == 1:
                axes = [axes]
            for ax, (time, image) in zip(axes, image_list):
                ax.imshow(image)
                ax.set_title(f"Time: {time}")
                ax.axis('off')
            plt.show()

    def get_tiles(self):
        return self.fetched_tiles


def RGB2LAB2(R0, G0, B0):
    R = R0 / 255
    G = G0 / 255
    B = B0 / 255
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    X = 0.449 * R + 0.353 * G + 0.198 * B
    Z = 0.012 * R + 0.089 * G + 0.899 * B
    L = Y
    a = (X - Y) / 0.234
    b = (Y - Z) / 0.785
    return L, a, b


class ConvertRGBToFeedModel(torch.nn.Module):
    def forward(self, img):
        img = np.asanyarray(img)
        sz_x = img.shape[0]
        sz_y = img.shape[1]
        train_imgs = np.zeros((sz_x, sz_y, 2))
        train_input = np.zeros((sz_x, sz_y, 1))
        R1 = np.reshape(img[:, :, 0], (sz_x * sz_y, 1))
        G1 = np.reshape(img[:, :, 1], (sz_x * sz_y, 1))
        B1 = np.reshape(img[:, :, 2], (sz_x * sz_y, 1))
        L, A, B = RGB2LAB2(R1, G1, B1)
        train_input[:, :, 0] = L.reshape((sz_x, sz_y))
        train_imgs[:, :, 0] = np.reshape(A, (sz_x, sz_y))
        train_imgs[:, :, 1] = np.reshape(B, (sz_x, sz_y))
        return (train_input, train_imgs)


def create_grayscale_tiles(fetched_tiles):
    converter = ConvertRGBToFeedModel()
    grayscale_tiles = {}

    for (longitude, latitude), image_list in fetched_tiles.items():
        grayscale_tiles[(longitude, latitude)] = []
        for time, image in image_list:
            L, _ = converter.forward(image)
            grayscale_image = Image.fromarray((L[:, :, 0] * 255).astype(np.uint8))
            grayscale_tiles[(longitude, latitude)].append((time, grayscale_image))

    return grayscale_tiles