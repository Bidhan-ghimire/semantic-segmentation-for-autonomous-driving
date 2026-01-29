# Semantic Segmentation for Autonomous Driving

This repository contains semantic segmentation experiments using **DeepLabV3+** (TensorFlow/Keras) with **ResNet18** and **ResNet50** backbones. The notebooks cover training and evaluation on **CamVid** and **Cityscapes**.

This project was completed as a **capstone project** for the course **Introduction to Deep Learning** taught by **Dr. Stephen Scott** at the **University of Nebraska-Lincoln (UNL)** in **Spring 2022**.

# Please find the report in docs folder.


# Notes on image-classifiers

This project uses classification_models.tfkeras, which comes from the image-classifiers package.

The provided requirements.txt pins image-classifiers==0.2.0.

If you encounter import/version issues on your machine, try image-classifiers==1.0.0b1 instead.

Datasets

Datasets are not included in this repository. You must download them separately and configure paths in the notebooks.

CamVid

Place CamVid images and labels on your machine.

Update the dataset paths at the top of the CamVid notebooks to match your local folder layout.

The CamVid pipeline expects a labels_color.txt file that maps RGB colors to class IDs.

Cityscapes

Cityscapes requires registration and manual download.

The notebooks load Cityscapes via tensorflow_datasets (tfds). Follow the instructions inside the Cityscapes notebook to set up the dataset locally.

src/datasets/labels.py is required for label ID to trainId mapping (the notebook imports labels and id2label from it). Include proper attribution for that file.

Model details

Architecture: DeepLabV3+

Backbones supported:

ResNet18

ResNet50

Model inputs:

image_size = [height, width]

num_classes (number of segmentation classes)

Mask format:

Masks are integer class IDs per pixel (sparse masks).

One-hot encoding is NOT required.

Training defaults

Maximum epochs: 100

Early stopping patience: 4

Notebooks

notebooks/01_prepare_camvid.ipynb
Preprocess CamVid, convert RGB masks to class IDs, optional augmentation, and save arrays if needed.

notebooks/02_train_camvid.ipynb
Train DeepLabV3+ on CamVid.

notebooks/03_evaluate_camvid.ipynb
Evaluate a trained model (accuracy, MeanIoU, confusion matrix) and visualize predictions.

notebooks/04_cityscapes_experiments.ipynb
Train/evaluate on Cityscapes using TFDS at different resolutions and backbones.

Report

The project report is available at:

docs/report.pdf



