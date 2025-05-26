import os
from yolov11 import YOLO
from PIL import Image

MODEL_PATH = 'runs/train/InfraOD_model/weights/best.pt'
IMAGES_DIR = 'test_images'  # Change this to your test images folder
OUTPUT_DIR = 'predictions'

os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    for img_name in os.listdir(IMAGES_DIR):
        if img_name.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(IMAGES_DIR, img_name)
            img = Image.open(img_path)
            results = model(img)
            # Save prediction image
            results.save(os.path.join(OUTPUT_DIR, img_name))
            # Print results
            print(f"Predictions for {img_name}:")
            print(results) 