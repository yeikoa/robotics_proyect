import cv2
import numpy as np
import requests
from ultralytics import YOLO
from queue import PriorityQueue
import serial
import time

from utils.translate_words import Translate
# ------------ CONFIGURACIÓN --------------------
IP_CAM_URL   = "http://192.168.89.231:8080/video"
GRID_SIZE    = 20
SERVER_URL   = "http://localhost:5000/objetivos"
#SERIAL_PORT  = "COM3"
#BAUD_RATE    = 9600

translator = Translate()

model_objetivos = YOLO("yolov8n.pt")
custom_model    = YOLO("D:/robotics_proyect/entrenamiento_robot/robot_model/weights/best.pt")

cap = cv2.VideoCapture(IP_CAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
#time.sleep(2)
ARDUINO_URL = "http://192.168.89.79/recibir_ruta"  

def translate_objetives(lista):

    return translator.masive_translate(lista, language_og="auto", language_des="en")

def obtener_objetivos():
    try:
        response = requests.get(SERVER_URL, timeout=2)
        if response.status_code == 200:
            data = response.json()
            lista_español = data.get("objetos", [])
            return translate_objetives(lista_español)
    except:
        pass
    return []

def get_grid_pos(frame_shape, x, y):
    h, w = frame_shape[:2]
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE
    col = x // cell_w
    row = y // cell_h
    return (row, col)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, grid):
    h, w = grid.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        current = open_set.get()[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dy, dx in [(0,1), (1,0), (0,-1), (-1,0)]:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))
    return []

objetivos = obtener_objetivos()
print("🎯 Objetivos recibidos (traducidos al inglés):", objetivos)
last_fetch_time = time.time()
FETCH_INTERVAL = 1  


ruta_completa = []
ruta_imagen = None
robot_pos = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    current_time = time.time()
    if current_time - last_fetch_time > FETCH_INTERVAL:
        nuevos_objetivos = obtener_objetivos()
        if nuevos_objetivos != objetivos:
            objetivos = nuevos_objetivos
            print("🔄 Objetivos actualizados:", objetivos)
        last_fetch_time = current_time
    # detectar posición del robot usando el modelo custom
    results_robot = custom_model(frame)
    robot_pos = None
    frame_drawn = frame.copy()

    for box in results_robot[0].boxes:
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        robot_pos = get_grid_pos(frame.shape, cx, cy)
        # Dibujar rectángulo amarillo alrededor del robot
        cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame_drawn,
                    "robot",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2)
        break  

    if robot_pos is None:
        cv2.imshow("Mapa y Ruta", frame_drawn)
        if cv2.waitKey(1) == 27:
            break
        continue

    results = model_objetivos(frame)

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    h, w = frame.shape[:2]
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE

    for i in range(1, GRID_SIZE):
        cv2.line(frame_drawn, (0, i * cell_h), (w, i * cell_h), (50, 50, 50), 1)
        cv2.line(frame_drawn, (i * cell_w, 0), (i * cell_w, h), (50, 50, 50), 1)

    cv2.rectangle(
        frame_drawn,
        (robot_pos[1] * cell_w, robot_pos[0] * cell_h),
        (robot_pos[1] * cell_w + cell_w, robot_pos[0] * cell_h + cell_h),
        (255, 255, 0),
        2
    )

    objetivos_pos = {}
    for r in results:
        for box in r.boxes:
            cls   = int(box.cls[0])
            label = model_objetivos.names[cls].lower()
            for objetivo in objetivos:
                if objetivo in label:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    gx, gy = get_grid_pos(frame.shape, cx, cy)
                    if objetivo not in objetivos_pos:
                        objetivos_pos[objetivo] = (gx, gy)
                        cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame_drawn,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
                    break

    actual_pos = robot_pos
    ruta_total = []

    for obj in objetivos:
        if obj in objetivos_pos:
            destino = objetivos_pos[obj]
            path = a_star(actual_pos, destino, grid)
            ruta_total += path
            actual_pos = destino

    if actual_pos != robot_pos:
        ruta_total += a_star(actual_pos, robot_pos, grid)

    # dibujar toda la ruta en rojo
    for gx, gy in ruta_total:
        x = gy * cell_w + cell_w // 2
        y = gx * cell_h + cell_h // 2
        cv2.circle(frame_drawn, (x, y), 5, (0, 0, 255), -1)

    # Si existe ruta entonces guardarla y enviarla por serial
    if ruta_total:
        ruta_completa = ruta_total.copy()
        ruta_imagen = frame_drawn.copy()

        with open("ruta_coordenadas.txt", "w") as f:
            for punto in ruta_completa:
                f.write(f"{punto}\n")
        cv2.imwrite("ruta_dibujada.png", ruta_imagen)
        print("📦 Ruta guardada como imagen y coordenadas.")

        try:
            # Construir datos como lista de diccionarios o pares
            coordenadas_json = {"ruta": ruta_completa}
            response = requests.post(ARDUINO_URL, json=coordenadas_json, timeout=3)
            if response.status_code == 200:
                print("📤 Ruta enviada exitosamente al robot por WiFi.")
            else:
                print(f"⚠️ Error al enviar ruta: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ No se pudo enviar la ruta al robot: {e}")

    cv2.imshow("Mapa y Ruta", frame_drawn)
    if cv2.waitKey(1) == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
