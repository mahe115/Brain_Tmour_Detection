# Brain Tumor Detection System

The Brain Tumor Detection System is a deep learning project aimed at classifying MRI images into four categories: No Tumor, Glioma Tumor, Meningioma Tumor, and Pituitary Tumor. This system utilizes a Convolutional Neural Network (CNN) trained on a balanced dataset of 44,000 MRI images, ensuring accurate and unbiased predictions across all categories.

## Project Overview

This project is developed to assist in the early detection and classification of brain tumors from MRI scans, which is crucial for treatment planning and improving patient outcomes. The CNN model has been trained with 11,000 images for each class, covering a wide range of tumor appearances and locations.

## Features

- Web application for easy interaction.
- Upload MRI images for instant tumor detection.
- Classification among four categories: No Tumor, Glioma, Meningioma, and Pituitary tumors.
- Built with Streamlit, enabling a smooth user experience.

## Technologies Used

- Python
- PyTorch
- Streamlit
- PIL (Python Imaging Library)
- Numpy

## Getting Started

### Prerequisites

Before running this project, ensure you have the following installed:

- Python (3.6 or later)
- torch
- streamlit
- numpy
- PIL

## Loss Function Graph

Below is the loss function graph obtained during the training of our model:

![Loss Function Graph](https://github.com/mahe115/Brain_Tmour_Detection/blob/14783e47e83804c7ef7e44f9af8800cfd14da1bd/0ce2c175-cc76-4cff-8f4d-17294d833bfc.jpg)

## Model Result in Web Interface Screenshots

Here are some snapshots of our web application interface:

![Interface 1](https://github.com/mahe115/Brain_Tmour_Detection/blob/14783e47e83804c7ef7e44f9af8800cfd14da1bd/4c06b083-3c69-4a6e-8aa1-740dbffba7ba.jpg)
![Interface 2](https://github.com/mahe115/Brain_Tmour_Detection/blob/14783e47e83804c7ef7e44f9af8800cfd14da1bd/54466196-5966-47a8-b09e-54f8974ff5b8.jpg)
![Interface 3](https://github.com/mahe115/Brain_Tmour_Detection/blob/14783e47e83804c7ef7e44f9af8800cfd14da1bd/c59c9032-40a2-4f47-91cf-42dfbf80eb53.jpg)

## Challenges

- **Data Preprocessing**: Ensuring the MRI images were of a consistent format and quality for training posed initial challenges.
- **Overfitting**: Given the complexity of the model and the diverse dataset, avoiding overfitting required careful tuning of the model parameters.

## Future Work

- Implementing additional layers or alternative architectures (like ResNet or Inception) to improve accuracy.
- Expanding the dataset with more diverse examples to further enhance the model's robustness.
- Developing a mobile application to make the system more accessible.

## Model Training

The CNN model was trained using a comprehensive notebook on Kaggle, which outlines the entire process, including data preprocessing, model architecture setup, training, and evaluation. You can access the notebook and delve into the specifics of the training process here:

[Kaggle Notebook for Brain Tumor Detection System Training](https://www.kaggle.com/code/mahendranb7/brain-tumour-classification?rvi=1)

