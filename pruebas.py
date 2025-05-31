import cv2
from ultralytics import YOLO

IP_CAM_URL = "http://10.251.48.176:8080/video"

# Carga el modelo YOLOv8 (nano para velocidad, puedes usar yolov8s.pt para mejor precisión)
model = YOLO("yolov8n.pt")

# Conecta con la cámara IP del celular
cap = cv2.VideoCapture(IP_CAM_URL)

if not cap.isOpened():
    print("❌ No se pudo conectar con la cámara. Revisa la IP.")
    exit()

print("✅ Conectado a la cámara. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame. Reintentando...")
        continue

    # Realiza detección con YOLOv8
    results = model(frame)

    # Dibuja resultados sobre el frame
    annotated_frame = results[0].plot()

    # Muestra la imagen con los objetos detectados
    cv2.imshow("🧠 Detección YOLOv8 desde celular", annotated_frame)

    if cv2.waitKey(1) == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()