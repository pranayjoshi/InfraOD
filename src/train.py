from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolo11n.pt')
    results = model.train(
        data='dataset.yaml',
        imgsz=336,
        epochs=50,
        batch=16,
        name='InfraOD_model'
    )
    print("Training complete. Results:", results)
