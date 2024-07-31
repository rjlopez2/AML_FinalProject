# COLORIZING SWITZERLAND: APPLYING DEEP LEARNING ON HISTORICAL AERIAL IMAGES

Final Project for the CAS in Advanced Machine Learning (UniBern 2023-2024)

Authors:

Lorenz Joss and Ruben Lopez

## About

Image colorization techniques aim to transform a grayscale image by assigning colors, so it becomes visually similar to our perceived reality. This task is challenging since a number of features need to be taken into account: global color richness balance, visual harmony, the conformity to object-level semantics, etc. while minimizing the visual artifacts such as color leakage or incomplete colorization. Current advances in machine learning (ML) and artificial intelligence (AI) have provided a huge contribution to automate this task. Historical aerial images is one of the many expertise domains that can profit from deep-learning base colorization applications. In this project we aim to use an existing model based on U-Net architecture and retrain it with historical aerial images from Switzerland.

![](/media/Preview_results.jpg)
Example of image colorization results using diferent models.

## Structrue of this repository

This repositry contain the script we used to perform the following task

- [data retrieval](scripts/retrieve_data_clean.ipynb)
- [data preparation](scripts/preprocessing_tf.ipynb)
- [Model training](scripts/UBELIX) usign containers to the HPC UBELIX
- [Model metrics](scripts/compute_models_metrics.py)  and [history plots](scripts/plot_model_history.ipynb)
- Model inference with color or grey images

## Data and Models weights availability

The ready-to-feed-model data and model's weights from this project can be found in the following links:

- [Data](https://perritos.myasustor.com:1986/data/)
- [Models](https://perritos.myasustor.com:1986/Models/)

If you wish to look at details on how the data was preprocessd before preparing and feed it to the models, please see the script for [data preparation](scripts/retrieve_data_clean.ipynb).

## Report issue

Please report issues or comments via [GitHub Issues](https://github.com/rjlopez2/AML_FinalProject/issues/new).


