import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os


# Set wide layout for full width
st.set_page_config(page_title="Brain Tumor Detection", layout="wide", page_icon="🧠")

# Custom CSS for spacing and class styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    .stTitle, .stHeader, .stSubheader, .stMarkdown, .stImage, .stButton, .stFileUploader, .stAlert {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }

    .css-1kyxreq {
        row-gap: 0.2rem !important;
    }

    div.stButton > button {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }

    .center-title h1 {
        text-align: center !important;
    }

    .result-box {
        padding: 0.5rem;
        border-radius: 10px;
        font-weight: bold;
        color: white;
        display: inline-block;
        margin-top: 0.5rem;
        text-align: center;
    }

    .no-tumor {
        background-color: rgba(0, 128, 0, 0.6); /* Green */
    }

    .tumor {
        background-color: rgba(255, 0, 0, 0.6); /* Red */
    }
    </style>
""", unsafe_allow_html=True)

# Centered title
st.markdown('<div class="center-title"><h1>Brain Tumor Detection System</h1></div>', unsafe_allow_html=True)

# Load model
model = YOLO("Brain_tumour_models1.pt")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI Image of the Brain...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_image_path = temp_file.name
        image.save(temp_image_path)

    results = model(temp_image_path, imgsz=640, conf=0.25)

    if st.button("Click For Result"):
        for result in results:
            annotated_img = result.plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Uploaded Image")
                st.image(image, width=image.width // 1)

            with col2:
                st.markdown("### Detection Output")
                st.image(annotated_img, width=annotated_img.shape[1] // 1)

                # Display detection result below the output image
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls_id].capitalize()
                        accuracy = f"{conf * 100:.2f}%"
                        css_class = "no-tumor" if label.lower() == "no tumor" else "tumor"
                        html = f'<div class="result-box {css_class}">Detected: <strong>{label}</strong> with accuracy: <strong>{accuracy}</strong></div>'
                        st.markdown(html, unsafe_allow_html=True)
                else:
                    html = '<div class="result-box no-tumor">No tumor detected in the uploaded MRI image.</div>'
                    st.markdown(html, unsafe_allow_html=True)

    os.remove(temp_image_path)
else:
    st.info("📤 Please upload a valid MRI image.")
