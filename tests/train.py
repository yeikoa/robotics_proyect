from ultralytics import YOLO


model = YOLO("yolov8n.pt")  

model.train(
    data="C:/yolo_train/mejorado/data.yaml", 
    epochs=50,
    imgsz=640,
    batch=8,
    project="entrenamiento_robot",
    name="robot_model"
)
