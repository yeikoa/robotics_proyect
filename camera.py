import cv2
import numpy as np
import requests
from ultralytics import YOLO
from queue import PriorityQueue
import serial
import time

from utils.translate_words import Translate

# ------------ CONFIGURACIÓN --------------------
IP_CAM_URL   = "http://172.17.44.237:8080/video"
GRID_SIZE    = 20
SERVER_URL   = "http://localhost:5000/objetivos"
SERIAL_PORT  = "COM6"  
BAUD_RATE    = 9600

translator = Translate()

model_objetivos = YOLO("yolov8n.pt")
custom_model    = YOLO("C:/Users/User/OneDrive - Universidad Nacional de Costa Rica/Escritorio/Programacion/Python/robotics_proyect/entrenamiento_robot/robot_model/weights/best.pt")

cap = cv2.VideoCapture(IP_CAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

# ------------ VARIABLES DE CONTROL --------------------
comandos_enviados = False  # Flag para controlar si ya se enviaron comandos

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
    """
    Convierte una ruta de coordenadas grid a instrucciones para el robot.
    
    SISTEMA DE ORIENTACIÓN DEL ROBOT:
    - Robot siempre orientado físicamente hacia "down"
    - down: se mueve en eje -x a +x (hacia la derecha en el plano)
    - left (1 giro): se orienta para ir de -y a +y (hacia arriba en el plano)
    - right (1 giro): se orienta para ir de +y a -y (hacia abajo en el plano)  
    - up (2 giros): se orienta para ir de +x a -x (hacia la izquierda en el plano)
    """
    if len(path) < 2:
        return []

    instrucciones = []

    # Mapeo de direcciones en el grid
    direcciones_grid = {
        (0, 1): "grid_right",   # Moverse hacia la derecha en el grid (+y)
        (0, -1): "grid_left",   # Moverse hacia la izquierda en el grid (-y)
        (1, 0): "grid_down",    # Moverse hacia abajo en el grid (+x)
        (-1, 0): "grid_up"      # Moverse hacia arriba en el grid (-x)
    }

    # Estado actual de orientación del robot
    # down = eje -x a +x, left = eje -y a +y, right = eje +y a -y, up = eje +x a -x
    orientacion_actual = "down"  # Robot inicia orientado hacia down
    
    def obtener_giros_necesarios(orientacion_actual, direccion_grid_deseada):
        """
        Calcula los giros necesarios para que el robot se oriente hacia la dirección deseada
        """
        
        # Mapeo: orientación del robot -> dirección que puede tomar en el grid
        orientacion_a_direccion = {
            "down": "grid_right",   # -x a +x = moverse hacia la derecha del grid
            "left": "grid_up",      # -y a +y = moverse hacia arriba del grid
            "right": "grid_down",   # +y a -y = moverse hacia abajo del grid  
            "up": "grid_left"       # +x a -x = moverse hacia la izquierda del grid
        }
        
        # Si ya está en la orientación correcta, no necesita girar
        if orientacion_a_direccion[orientacion_actual] == direccion_grid_deseada:
            return [], orientacion_actual
        
        # Calcular giros necesarios
        if direccion_grid_deseada == "grid_right":  # Necesita orientación "down"
            if orientacion_actual == "down":
                return [], "down"
            elif orientacion_actual == "left":
                return ["right"], "down"  # 1 giro derecha
            elif orientacion_actual == "right":
                return ["left"], "down"   # 1 giro izquierda
            elif orientacion_actual == "up":
                return ["right", "right"], "down"  # 2 giros = media vuelta
                
        elif direccion_grid_deseada == "grid_up":   # Necesita orientación "left"
            if orientacion_actual == "down":
                return ["left"], "left"   # 1 giro izquierda
            elif orientacion_actual == "left":
                return [], "left"
            elif orientacion_actual == "right":
                return ["right", "right"], "left"  # 2 giros = media vuelta
            elif orientacion_actual == "up":
                return ["right"], "left"  # 1 giro derecha
                
        elif direccion_grid_deseada == "grid_down": # Necesita orientación "right"
            if orientacion_actual == "down":
                return ["right"], "right" # 1 giro derecha
            elif orientacion_actual == "left":
                return ["right", "right"], "right"  # 2 giros = media vuelta
            elif orientacion_actual == "right":
                return [], "right"
            elif orientacion_actual == "up":
                return ["left"], "right"  # 1 giro izquierda
                
        elif direccion_grid_deseada == "grid_left": # Necesita orientación "up"
            if orientacion_actual == "down":
                return ["right", "right"], "up"  # 2 giros = media vuelta
            elif orientacion_actual == "left":
                return ["right"], "up"    # 1 giro derecha
            elif orientacion_actual == "right":
                return ["left"], "up"     # 1 giro izquierda
            elif orientacion_actual == "up":
                return [], "up"
        
        return [], orientacion_actual  # Fallback

    # ✅ Procesar cada movimiento en la ruta
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]  # Cambio en x (filas)
        dy = path[i][1] - path[i-1][1]  # Cambio en y (columnas)
        direccion_grid = direcciones_grid.get((dx, dy))
        
        if direccion_grid is None:
            continue
        
        print(f"🎯 Movimiento {i}: {path[i-1]} -> {path[i]} = {direccion_grid}")
        print(f"   Orientación actual del robot: {orientacion_actual}")
        
        # Calcular giros necesarios y nueva orientación
        giros, nueva_orientacion = obtener_giros_necesarios(orientacion_actual, direccion_grid)
        
        # Agregar giros a las instrucciones
        if giros:
            instrucciones.extend(giros)
            print(f"   🔄 Giros necesarios: {giros}")
        
        # Actualizar orientación
        orientacion_actual = nueva_orientacion
        print(f"   Nueva orientación: {orientacion_actual}")
        
        # Siempre avanzar con "down" (comando físico del robot)
        instrucciones.append("down")
        print(f"   ➡️ Avanzar con 'down'")
    
    return instrucciones

def enviar_comandos_una_vez(ruta_actual):
    """
    Envía comandos al robot solo una vez cuando se define la ruta
    """
    global comandos_enviados
    
    if not comandos_enviados and ruta_actual:  # Solo si no se han enviado y hay ruta
        print("\n🚀 ENVIANDO COMANDOS AL ROBOT...")
        
        # Guardar archivos de ruta
        with open("ruta_coordenadas.txt", "w") as f:
            for punto in ruta_actual:
                f.write(f"{punto}\n")
        
        # Convertir ruta a instrucciones
        instrucciones = convertir_a_instrucciones(ruta_actual)
        
        if instrucciones:  # Solo enviar si hay instrucciones
            print("➡️ Instrucciones enviadas:")
            for inst in instrucciones:
                print(f"  {inst}")
                ser.write(f"{inst}\n".encode())
                time.sleep(0.1)
            
            ser.write(b"END\n")
            print("📤 Fin de ruta enviada al robot.")
            
            # Marcar como enviados
            comandos_enviados = True
            return True  # Indica que se enviaron comandos
        else:
            print("⚠️ No se generaron instrucciones válidas")
    
    return False  # No se enviaron comandos

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

    # Dibujar la ruta en el frame
    for gx, gy in ruta_total:
        x = gy * cell_w + cell_w // 2
        y = gx * cell_h + cell_h // 2
        cv2.circle(frame_drawn, (x, y), 5, (0, 0, 255), -1)

    # 🔥 Enviar comandos solo UNA vez cuando se define la ruta
    if ruta_total:
        comandos_enviados_ahora = enviar_comandos_una_vez(ruta_total)
        
        if comandos_enviados_ahora:
            # Guardar imagen solo cuando se envían comandos
            cv2.imwrite("ruta_dibujada.png", frame_drawn)
            print("📦 Ruta guardada como imagen.")

    # Mostrar estado en la imagen
    status_text = "COMANDOS ENVIADOS" if comandos_enviados else "CALCULANDO RUTA..."
    cv2.putText(frame_drawn, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 0) if comandos_enviados else (0, 255, 255), 2)

    cv2.imshow("Mapa y Ruta", frame_drawn)
    
    # ESC para salir
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()