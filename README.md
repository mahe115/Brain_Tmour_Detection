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

### Installation

Clone this repository to your local machine:

```bash
https://github.com/mahe115/Brain_Tmour_Detection.git
Navigate to the project directory:

cd brain-tumor-detection-system
Install the required dependencies:

pip install -r requirements.txt
Running the Application
To start the Streamlit web application, run the following command in your terminal:

streamlit run app.py
Navigate to the localhost URL provided by Streamlit to interact with the application.

Usage
Once the application is running, select the 'Brain Tumor Detection' option.
Upload an MRI image of the brain using the file uploader.
Click "Click For Result" to classify the image.
The application will display the classification result, identifying the presence and type of tumor if applicable.
Model Details
The CNN model architecture consists of sequential convolutional layers, max pooling, and linear layers, culminating in a classification among the four possible outcomes. For detailed architecture and training process, refer to the Kaggle Notebook.

Contributions
Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

vbnet
Copy code

Remember to replace `<your-github-username>` with your actual GitHub username where indicated. Thi
