# 🧠 Brain Tumor Detection System (YOLO-based)

This project is an upgraded version of the Brain Tumor Detection System, now powered by the **YOLO object detection model** for improved performance and real-time tumor localization from MRI images. The new system offers an intuitive Streamlit-based web interface that allows users to upload brain MRI scans and receive instant tumor detection results.

## 🚀 Project Overview

Brain tumors can be life-threatening if not diagnosed early. This application aims to assist radiologists and medical practitioners by automating the detection of brain tumors from MRI scans using the **YOLOv8 model**. The web application highlights tumor regions and provides classification feedback with detection accuracy.

## 🎯 Key Features

- ✅ Upload MRI brain images (`.jpg`, `.jpeg`, `.png`)
- ✅ Real-time object detection using a fine-tuned YOLOv8 model
- ✅ Visual feedback with bounding boxes and class confidence
- ✅ Clean and modern Streamlit web interface
- ✅ Instant classification: "No Tumor", "Glioma", "Meningioma", or "Pituitary Tumor"
## Sample Web-Interface

 ![Sample Web-Interface](https://github.com/mahe115/Brain_Tumour_Detection/blob/8c29cbccf6234ee22f0172cd60b466ab11583cba/Perfromance%20and%20output%20images/Screenshot%20(404).png)

 ![Predictions](https://github.com/mahe115/Brain_Tumour_Detection/blob/1d612dadcd7c5c3e92f007cc8a26d52d0e48b47b/Perfromance%20and%20output%20images/val_batch0_labels.jpg)


## 🧠 Technologies Used

- Python
- [YOLOv8](https://github.com/ultralytics/ultralytics) (via `ultralytics` package)
- Streamlit
- OpenCV
- PIL (Python Imaging Library)
- Tempfile and OS modules for secure file handling

## 🖥️ Interface Preview

Upon uploading an image and clicking **"Click For Result"**, the application displays:
- The original image
- The annotated image with tumor detection (if present)
- A styled result box indicating tumor type and detection accuracy
## Performance Graph

Below is Performance  graph details obtained during the training of our model:

![Performance Graph](https://github.com/mahe115/Brain_Tumour_Detection/blob/e577bdf8ee371298d3aacfc0f535d667a01383c5/Perfromance%20and%20output%20images/results.png)

## Example Output:
- Detected: Glioma with accuracy: 82.24%

![Loss Function Graph](https://github.com/mahe115/Brain_Tumour_Detection/blob/8c29cbccf6234ee22f0172cd60b466ab11583cba/Perfromance%20and%20output%20images/Screenshot%20(405).png)


- No tumor detected in the uploaded MRI image.

![Loss Function Graph](https://github.com/mahe115/Brain_Tumour_Detection/blob/8c29cbccf6234ee22f0172cd60b466ab11583cba/Perfromance%20and%20output%20images/Screenshot%20(406).png)



## ⚠️ Challenges

- **Data Preprocessing**: MRI image formats and resolutions varied across the dataset. Consistent resizing and annotation formatting (YOLO label format) were crucial for optimal training.
- **Annotation Accuracy**: Manual annotation for YOLO format was time-consuming and prone to error, which required verification and cleaning for model consistency.
- **Class Imbalance**: Some tumor classes were underrepresented, leading to class imbalance issues that required augmentation strategies.
- **Real-Time Optimization**: Ensuring fast and accurate detection without compromising on model size was critical for web deployment via Streamlit.

## 🔮 Future Work

- Integrating support for **DICOM** image processing to enable direct use of medical-grade scan data.
- Exploring **YOLOv8-seg** or other segmentation models to not only detect but **segment tumor regions more precisely**.
- Creating a **cross-platform mobile app** using tools like React Native or Flutter for on-the-go diagnostics.
- Deploying the model with **ONNX/TensorRT** for faster inference and reduced latency in real-time environments.
- Adding **explainability** with Grad-CAM-like visualizations for better clinical acceptance.

## 🧠 Model Training (YOLOv8)

The YOLOv8 model was trained using a curated and annotated dataset of MRI brain images. Key steps in the training process included:

- Image augmentation (flips, rotation, contrast enhancement)
- YOLO-format labeling (`.txt` files with bounding boxes and class labels)
- Model fine-tuning with:
  - Epochs: 50
  - Input size: 640x640
  - Batch size: 16
  - Optimizer: SGD
  - Learning rate scheduler for dynamic learning rates

Training was conducted using Ultralytics’ YOLOv8 framework with GPU acceleration. The model achieved over **98% accuracy** in detecting and classifying tumors like **Glioma, Meningioma, Pituitary Tumor**, and **No Tumor**.

> **Access the earlier CNN-based training notebook for reference or comparison**:  
> [Kaggle Notebook - Brain Tumor Classification (CNN Approach)](https://www.kaggle.com/code/mahendranb7/brain-tumour-classification?rvi=1)
