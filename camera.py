import cv2
import numpy as np
import requests
from ultralytics import YOLO
from queue import PriorityQueue
import serial
import time

from utils.translate_words import Translate

# ------------ CONFIGURACIÓN --------------------
IP_CAM_URL   = "http://10.251.48.124:8080/video"
GRID_SIZE    = 20
SERVER_URL   = "http://localhost:5000/objetivos"
SERIAL_PORT  = "COM4"
BAUD_RATE    = 9600

translator = Translate()

model_objetivos = YOLO("yolov8n.pt")
custom_model    = YOLO("D:/robotics_proyect/entrenamiento_robot/robot_model/weights/best.pt")

cap = cv2.VideoCapture(IP_CAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

# ------------ FUNCIONES --------------------

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

def eliminar_repetidos(ruta):
    return [p for i, p in enumerate(ruta) if i == 0 or p != ruta[i-1]]

def convertir_a_instrucciones(path):
    if len(path) < 2:
        return []

    instrucciones = []
    direccion_actual = "down"  # dirección inicial

    direcciones = {
        (0, 1): "right",
        (0, -1): "left",
        (1, 0): "down",
        (-1, 0): "up"
    }

    orientaciones = ["up", "right", "down", "left"]

    def calcular_giro(d_actual, d_nueva):
        idx_a = orientaciones.index(d_actual)
        idx_b = orientaciones.index(d_nueva)
        delta = (idx_b - idx_a) % 4
        if delta == 1:
            return "right"
        elif delta == 3:
            return "left"
        elif delta == 2:
            return ["right", "right"]
        return None

    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        nueva_direccion = direcciones.get((dx, dy))
        if nueva_direccion is None:
            continue

        if nueva_direccion != direccion_actual:
            giro = calcular_giro(direccion_actual, nueva_direccion)
            if isinstance(giro, list):
                instrucciones.extend(giro)
            elif giro:
                instrucciones.append(giro)
            direccion_actual = nueva_direccion

        instrucciones.append("down")  # avanzar después de orientar

    return instrucciones

# ------------ LOOP PRINCIPAL --------------------

objetivos = obtener_objetivos()
print("🎯 Objetivos recibidos (traducidos al inglés):", objetivos)
last_fetch_time = time.time()
FETCH_INTERVAL = 1

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

    results_robot = custom_model(frame)
    robot_pos = None
    frame_drawn = frame.copy()

    for box in results_robot[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        offset = 5
        cx = x2 + offset
        cy = (y1 + y2) // 2
        robot_pos = get_grid_pos(frame.shape, cx, cy)

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
            cls = int(box.cls[0])
            label = model_objetivos.names[cls].lower()
            for objetivo in objetivos:
                if objetivo in label:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    offset = 5
                    cx = x1 - offset
                    cy = (y1 + y2) // 2
                    gx, gy = get_grid_pos(frame.shape, cx, cy)

                    if objetivo not in objetivos_pos:
                        objetivos_pos[objetivo] = (gx, gy)
                        cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_drawn, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break

    actual_pos = robot_pos
    ruta_total = []

    for obj in objetivos:
        if obj in objetivos_pos:
            destino = objetivos_pos[obj]
            path = a_star(actual_pos, destino, grid)
            ruta_total += path
            actual_pos = destino

    ruta_total = eliminar_repetidos(ruta_total)

    for gx, gy in ruta_total:
        x = gy * cell_w + cell_w // 2
        y = gx * cell_h + cell_h // 2
        cv2.circle(frame_drawn, (x, y), 5, (0, 0, 255), -1)

    if ruta_total:
        with open("ruta_coordenadas.txt", "w") as f:
            for punto in ruta_total:
                f.write(f"{punto}\n")
        cv2.imwrite("ruta_dibujada.png", frame_drawn)
        print("📦 Ruta guardada como imagen y coordenadas.")

        instrucciones = convertir_a_instrucciones(ruta_total)
        print("➡️ Instrucciones enviadas:")
        for inst in instrucciones:
            print(f"  {inst}")
            ser.write(f"{inst}\n".encode())
            time.sleep(0.1)

        ser.write(b"END\n")
        print("📤 Fin de ruta enviada al robot.")

    cv2.imshow("Mapa y Ruta", frame_drawn)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
