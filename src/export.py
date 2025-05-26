from ultralytics import YOLO
import os

MODEL_PATH = 'runs/train/InfraOD_model/weights/best.pt'

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    result = model.export(format='onnx')
    print(f"Exported model to ONNX format: {result['file']}") 