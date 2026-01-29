# Semantic Segmentation for Autonomous Driving

This repository contains semantic segmentation experiments using **DeepLabV3+** (TensorFlow/Keras) with **ResNet18** and **ResNet50** backbones. The notebooks cover training and evaluation on **CamVid** and **Cityscapes**.

This project was completed as a **capstone project** for the course **Introduction to Deep Learning** taught by **Dr. Stephen Scott** at the **University of Nebraska-Lincoln (UNL)** in **Spring 2022**.




# Regarding the model:

1. Patience for early stopping = 4
2. Maximum number of epochs = 100
3. THE MODEL DOESNOT REQUIRE ONE-HOT ENCODDED IMAGE MASKS
4. The model needs as input image_size = [height,width] and number of classes
5. There are two models for each of resnet 18 and 50
