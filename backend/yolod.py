from ultralytics import YOLO

def download_yolov8s():
    # This automatically downloads the YOLOv8n model if not already present locally
    model = YOLO('yolov8l.pt')
    print("YOLOv8s model downloaded and loaded:")
    print(model)
    return model

if __name__ == "__main__":
    download_yolov8s()