
# Sistema de Navegación y Control de Robot

Este proyecto permite controlar un robot usando comandos de voz, procesamiento de lenguaje natural y visión por computadora. Con una interfaz sencilla en PyQt5, el usuario graba su voz, el sistema transcribe y entiende el mensaje, genera un plan de ruta hacia objetos detectados en una cámara, y finalmente envía instrucciones físicas al robot vía puerto serial.




## 🎯 Características

- Interfaz gráfica para grabar comandos de voz
- Transcripción automática con Whisper
- Interpretación semántica con Gemini (Google Generative AI)
- Comunicación con servidor Flask
- Traducción automática de objetivos con googletrans
- Detección de objetos y robot con YOLOv8
- Entrenamiento de modelo personalizado
- Planeación de rutas con algoritmo A*
- Control de robot vía puerto serial


## Screenshots
FALTA
![App Screenshot]()


## variables de entorno

Para iniciar este proyecto se necesita una llave de Google AI Studio, en .env.local se agrega esta.

`KEY_GEMINI` = "............."


## 🚀 Instalación
### Requisitos
- Python 3.8+
-  IP cam
- Arduino o microcontrolador conectado a puerto serial
### Instalación
```bash
#Clonar repositorio
 git clone https://github.com/yeikoa/robotics_proyect.git

# Instalar dependencias
pip install -r requirements.txt
```
## ⚙️ Uso
-  Ejecutar la interfaz de control por voz
```bash
python ./voice_detector.py
```
- Iniciar servidor desde esta interfaz y grabar el comando de voz para que se envíe a servidor
- Iniciar el codigo de python para la camara y capturar la ruta del robot y que se detecten los objetivos de este mediante IA
```bash
python ./camera.py
```
## 🛠Tecnologías usadas
- Python
- PyQt5
- Whisper
- Google Generative AI
- YOLOv8
- Flask
- OpenCV
- googletrans
- Roboflow

## Autores

- [@yeikoa](https://www.github.com/yeikoa)
- [@AxelOreamuno](https://www.github.com/AxelOreamuno)

