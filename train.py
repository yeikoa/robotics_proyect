from ultralytics import YOLO

# Cargar modelo base
model = YOLO("yolov8n.pt")  # Puedes usar yolov8s.pt, yolov8m.pt según tu hardware

# Entrenar
model.train(
    data="D:/yolo_train/proyecto.v1i.yolov5pytorch/data.yaml",  # Cambia esto por la ruta real
    epochs=50,
    imgsz=640,
    batch=8,
    project="entrenamiento_robot",
    name="robot_model"
)
