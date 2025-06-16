import cv2
from ultralytics import YOLO

IP_CAM_URL = "http://192.168.18.5:8080/video"

general_model = YOLO("yolov8m.pt")

custom_model = YOLO("C:/Trabajos_U/robotics_proyect/entrenamiento_robot/robot_model/weights/best.pt")  


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

   
    results_general = general_model(frame)

    results_custom = custom_model(frame)

    frame_general = results_general[0].plot()
    frame_final = results_custom[0].plot()  
 
    cv2.imshow("🧠 YOLOv8 - General + Robot", frame_final)

    if cv2.waitKey(1) == 27:  #  ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
