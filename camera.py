import cv2
import numpy as np
import requests
from ultralytics import YOLO
from queue import PriorityQueue
import serial
import time

# ------------ CONFIGURACIÓN --------------------
IP_CAM_URL = "http://10.251.48.176:8080/video"
GRID_SIZE = 20
SERVER_URL = "http://localhost:5000/objetivos"
SERIAL_PORT = "COM4"  
BAUD_RATE = 9600
TRADUCCION = {
    "caja azul": "blue box",
    "persona": "person",
    "silla": "chair",
    "botella": "bottle",
    "carro": "car",
    "maletin": "suitcase",
}

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(IP_CAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)  # Esperar conexión serial

def traducir_objetivos(lista):
    return [TRADUCCION.get(o.lower(), o.lower()) for o in lista]

def obtener_objetivos():
    try:
        response = requests.get(SERVER_URL, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return traducir_objetivos(data.get("objetos", []))
    except:
        pass
    return []

def get_grid_pos(frame_shape, x, y):
    h, w = frame_shape[:2]
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE
    return (y // cell_h, x // cell_w)

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
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            neighbor = (current[0]+dy, current[1]+dx)
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

#  Cargar objetivos
objetivos = obtener_objetivos()
print("🎯 Objetivos recibidos:", objetivos)

ruta_completa = []
ruta_imagen = None
robot_pos = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    frame_drawn = frame.copy()

    h, w = frame.shape[:2]
    cell_w = w // GRID_SIZE
    cell_h = h // GRID_SIZE

    for i in range(1, GRID_SIZE):
        cv2.line(frame_drawn, (0, i * cell_h), (w, i * cell_h), (50, 50, 50), 1)
        cv2.line(frame_drawn, (i * cell_w, 0), (i * cell_w, h), (50, 50, 50), 1)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            if "mouse" in label:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                robot_pos = get_grid_pos(frame.shape, cx, cy)
                cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (255,255,0), 2)
                cv2.putText(frame_drawn, "robot", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                break

    if robot_pos is None:
        cv2.imshow("Mapa y Ruta", frame_drawn)
        if cv2.waitKey(1) == 27:
            break
        continue

    cv2.rectangle(frame_drawn, (robot_pos[1]*cell_w, robot_pos[0]*cell_h),
                  (robot_pos[1]*cell_w + cell_w, robot_pos[0]*cell_h + cell_h), (255, 255, 0), 2)

    objetivos_pos = {}
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            for objetivo in objetivos:
                if objetivo in label:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    gx, gy = get_grid_pos(frame.shape, cx, cy)
                    if objetivo not in objetivos_pos:
                        objetivos_pos[objetivo] = (gx, gy)
                        cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame_drawn, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
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

    for gx, gy in ruta_total:
        x = gy * cell_w + cell_w // 2
        y = gx * cell_h + cell_h // 2
        cv2.circle(frame_drawn, (x, y), 5, (0, 0, 255), -1)

    if ruta_total:
        ruta_completa = ruta_total.copy()
        ruta_imagen = frame_drawn.copy()

        # Guardar archivos
        with open("ruta_coordenadas.txt", "w") as f:
            for punto in ruta_completa:
                f.write(f"{punto}\n")
        cv2.imwrite("ruta_dibujada.png", ruta_imagen)
        print("📦 Ruta guardada como imagen y coordenadas.")

        # Enviar por serial
        for punto in ruta_completa:
            ser.write(f"{punto[0]},{punto[1]}\n".encode())
            time.sleep(0.1)  # Espera para que el Arduino lea

        ser.write(b"END\n")
        print("📤 Ruta enviada al robot.")

    cv2.imshow("Mapa y Ruta", frame_drawn)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
