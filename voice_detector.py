import sys
import threading
import queue
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import google.generativeai as genai
import requests, json
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QLabel, QMessageBox, QHBoxLayout
)
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import Qt, QTimer

from take_keys import takeKeys

KEY = takeKeys()
genai.configure(api_key=KEY)

recording = False
audio_data_buffer = queue.Queue()
sample_rate = 44100
channels = 1

def callback(indata, frames, time, status):
    if status:
        print(status)
    if recording:
        audio_data_buffer.put(indata.copy())

def start_recording():
    global recording, audio_data_buffer
    recording = True
    audio_data_buffer = queue.Queue()
    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
            while recording:
                sd.sleep(100)
    except Exception as e:
        print(f"Error durante grabación: {e}")
    finally:
        join_recording = []
        while not audio_data_buffer.empty():
            join_recording.append(audio_data_buffer.get())

        if join_recording:
            final_audio = np.concatenate(join_recording, axis=0)
            wav.write("comando.wav", sample_rate, final_audio)

def stop_recording():
    global recording
    recording = False

def transcribir_audio(nombre_archivo="comando.wav", language="es"):
    try:
        modelo = whisper.load_model("medium")
        resultado = modelo.transcribe(nombre_archivo, language=language)
        return resultado["text"]
    except Exception as e:
        return f"Error al transcribir: {e}"

def interpretar_comando_con_gemini(texto_usuario):
    if not texto_usuario.strip():
        return "No hay texto para interpretar."

    prompt = f"""
Eres un asistente que interpreta comandos de voz para controlar un robot.
Convierte el siguiente mensaje en un JSON con este formato:
{{
  "accion": "ir",
  "objetos": ["objeto1", "objeto2", ...],
  "regresar": true/false
}}

Ejemplos:
- Entrada: "ve a la caja azul" → {{ "accion": "ir", "objetos": ["caja azul"], "regresar": false }}
- Entrada: "anda a la botella y al laptop, y luego vuelve" → {{ "accion": "ir", "objetos": ["botella", "laptop"], "regresar": true }}
- Entrada: "visita la caja y regresa después" → {{ "accion": "ir", "objetos": ["caja"], "regresar": true }}

Ten cuidado de interpretar correctamente las intenciones del usuario, incluso si hay errores gramaticales.

Mensaje: "{texto_usuario}"
"""

    model = genai.GenerativeModel('models/gemini-2.0-flash')
    try:
        respuesta = model.generate_content(prompt)
        return respuesta.text
    except Exception as e:
        return f"Error al interpretar: {e}"

class PulsatingCircle(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.setMaximumHeight(100)
        self.radius = 10
        self.growing = True
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.active = False

    def start(self):
        self.active = True
        self.timer.start(50)
        self.show()

    def stop(self):
        self.active = False
        self.timer.stop()
        self.hide()

    def animate(self):
        if self.growing:
            self.radius += 2
            if self.radius > 30:
                self.growing = False
        else:
            self.radius -= 2
            if self.radius < 10:
                self.growing = True
        self.update()

    def paintEvent(self, event):
        if not self.active:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()
        painter.setBrush(QBrush(QColor(255, 0, 0, 150)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, self.radius, self.radius)

class VoiceControlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 Control de Voz para Robot")
        self.setGeometry(100, 100, 650, 550)

        self.layout = QVBoxLayout()

        self.label_status = QLabel("Estado: Listo")
        self.circle_animation = PulsatingCircle()

        self.btn_start = QPushButton("🎙️ Iniciar Grabación")
        self.btn_stop = QPushButton("⏹️ Detener y Procesar")
        self.btn_clear = QPushButton("🧹 Limpiar Pantalla")
        self.btn_flask = QPushButton("🚀 Iniciar Servidor Flask")

        self.text_transcripcion = QTextEdit()
        self.text_comando = QTextEdit()

        self.text_transcripcion.setPlaceholderText("📝 Texto transcrito...")
        self.text_comando.setPlaceholderText("🤖 Comando interpretado (JSON)...")

        self.btn_start.clicked.connect(self.iniciar_grabacion)
        self.btn_stop.clicked.connect(self.detener_y_procesar)
        self.btn_clear.clicked.connect(self.limpiar_textos)
        self.btn_flask.clicked.connect(self.iniciar_servidor_flask)

        botones_layout = QHBoxLayout()
        botones_layout.addWidget(self.btn_start)
        botones_layout.addWidget(self.btn_stop)
        botones_layout.addWidget(self.btn_clear)
        botones_layout.addWidget(self.btn_flask)

        self.layout.addWidget(self.label_status)
        self.layout.addWidget(self.circle_animation)
        self.layout.addLayout(botones_layout)
        self.layout.addWidget(QLabel("Texto transcrito:"))
        self.layout.addWidget(self.text_transcripcion)
        self.layout.addWidget(QLabel("Comando interpretado:"))
        self.layout.addWidget(self.text_comando)

        self.setLayout(self.layout)
        self.hilo_grabacion = None

        self.flask_process = None
        self.verificar_status_timer = QTimer()
        self.verificar_status_timer.timeout.connect(self.verificar_status_servidor)
        self.verificar_status_timer.start(15000)  

    def iniciar_grabacion(self):
        if not recording:
            self.label_status.setText("Estado: Grabando...")
            self.circle_animation.start()
            self.hilo_grabacion = threading.Thread(target=start_recording)
            self.hilo_grabacion.start()
        else:
            QMessageBox.warning(self, "Advertencia", "Ya se está grabando.")

    def detener_y_procesar(self):
        global recording
        if recording:
            stop_recording()
            self.circle_animation.stop()
            if self.hilo_grabacion and self.hilo_grabacion.is_alive():
                self.hilo_grabacion.join()

            self.label_status.setText("Estado: Procesando audio...")
            texto = transcribir_audio("comando.wav")
            self.text_transcripcion.setPlainText(texto)

            comando = interpretar_comando_con_gemini(texto)

            comando = comando.strip()
            if comando.startswith("```"):
                comando = comando.replace("```json", "").replace("```", "").strip()
            if comando.startswith(";"):
                comando = comando[1:].strip()

            self.text_comando.setPlainText(comando)
            self.label_status.setText("Estado: Listo")

            try:
                json_comando = json.loads(comando)
                response = requests.post("http://localhost:5000/comando", json=json_comando)
                if response.status_code == 200:
                    print("✅ Comando enviado al servidor Flask.")
                else:
                    print(f"⚠️ Error al enviar comando: {response.status_code}")
            except json.JSONDecodeError as e:
                print("⚠️ La respuesta no es un JSON válido.")
                print(f"Detalle del error: {e}")
            except Exception as e:
                print(f"❌ Error al enviar el comando: {e}")
        else:
            QMessageBox.information(self, "Info", "No hay grabación activa.")

    def limpiar_textos(self):
        self.text_transcripcion.clear()
        self.text_comando.clear()
        self.label_status.setText("Estado: Listo")
        self.circle_animation.stop()

    def iniciar_servidor_flask(self):
        if not self.flask_process:
            self.flask_process = subprocess.Popen([
                sys.executable, "server.py" 
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.label_status.setText("Servidor Flask iniciado.")

    def verificar_status_servidor(self):
        try:
            r = requests.get("http://localhost:5000/status", timeout=1)
            if r.status_code == 200:
                print("✅ Servidor Flask activo.")
        except:
            print("❌ Servidor Flask no disponible.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VoiceControlApp()
    ventana.show()
    sys.exit(app.exec_())
