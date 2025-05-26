from ultralytics import YOLO
import cv2
import os
import glob

# Load the fine-tuned YOLO model
model = YOLO("Brain_tumour_models1.pt")

# Define the input folder path
input_folder = r"Val\No Tumor\images"  # Replace with your folder path

# Optional: Output folder to save annotated images
output_folder = "No_tumour_detections"
os.makedirs(output_folder, exist_ok=True)

# Get all image file paths (jpg, png, jpeg, etc.)
image_extensions = ["*.jpg", "*.png", "*.jpeg"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

# Run inference on each image
for img_path in image_paths:
    results = model(img_path, imgsz=640, conf=0.25)
    for i, result in enumerate(results):
        # Annotated image (with bounding boxes)
        annotated_img = result.plot()

        # Display the image
        cv2.imshow("Detection", annotated_img)
        cv2.waitKey(0)

        # Save annotated image
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, annotated_img)

cv2.destroyAllWindows()
