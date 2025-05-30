import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import google.generativeai as genai
import threading, queue, keyboard
import numpy as np
from take_keys import takeKeys

KEY = takeKeys()  
genai.configure(api_key= KEY)

recording = False
audio_data_buffer = queue.Queue()
sample_rate = 44100
channels = 1

def callback(indata, frames, time, status):
    if status:
        print(status)
    if recording:
        audio_data_buffer.put(indata.copy())

def start_recording(nombre_archivo):
    global recording, audio_data_buffer
    recording = True
    audio_data_buffer = queue.Queue()
    print("🎙️ Grabación iniciada. Presiona 's' para detener.")

    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback) as stream:
            while recording:
                sd.sleep(100)
    except Exception as e:
        print(f"❌Error durante la grabación: {e}")
    finally:
        join_recording = []
        while not audio_data_buffer.empty():
            join_recording.append(audio_data_buffer.get())

        if join_recording:  
            final_audio = np.concatenate(join_recording, axis=0)
            wav.write("comando.wav", sample_rate, final_audio)
            print("✅ Grabación detenida y guardada como 'comando.wav'.")      
        else:
            print("❌ No se grabó ningún audio.")
def stop_recording():
    global recording
    recording = False
    print("⏹️ Deteniendo grabación...")    


def transcribir_audio(nombre_archivo="comando.wav"):
    try:
        modelo = whisper.load_model("base")
        resultado = modelo.transcribe(nombre_archivo)
        texto = resultado["text"]
        print(f"📝 Texto reconocido: {texto}")
        return texto
    except Exception as e:
        print(f"❌ Error al transcribir el audio: {e}")
        return ""

def interpretar_comando_con_gemini(texto_usuario):
    if not texto_usuario.strip(): 
        print("🤔 No hay texto para interpretar.")
        return ""
    prompt = f"""
Eres un asistente que interpreta comandos de voz para controlar un robot.
Convierte el siguiente mensaje en un JSON claro. Ejemplo:
Entrada: "ve a la caja azul" → Salida: {{"accion": "ir", "objeto": "caja azul"}}
Ten en uenta que el mensaje puede contener errores asi que debes ser flexible en la interpretación
y mejorar el mensaje para que sea claro y conciso, ademas si ves palabras unidas o que posiblemente se tradujeron mal
mejoralas y responde.

Mensaje: "{texto_usuario}"
"""
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    try:
        respuesta = model.generate_content(prompt)
        print("🤖 Comando estructurado:")
        print(respuesta.text)
        return respuesta.text
    except Exception as e:
        print(f"❌ Error al interpretar el comando: {e}")
        print("Revisar la API KEY")
        return ""

# -------- PROGRAMA PRINCIPAL --------
def menu():
    print("🎤 Voice Detector - Control de Robot")
    print("Presiona 'r' para INICIAR la grabación.")
    print("Presiona 's' para DETENER la grabación.")
    print("Presiona 'q' para SALIR.")

    thread_recording = None
    while True:
        event = keyboard.read_event(suppress=True)
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'r':
                if not recording:
                    print("🔴 Iniciando grabación...")
                    thread_recording = threading.Thread(target=start_recording, args=("comando.wav",))
                    thread_recording.start()

                else:
                    print("⚠️ Ya se está grabando.")
              
            elif event.name == 's':
                if recording:
                    stop_recording()
                    if thread_recording and thread_recording.is_alive():
                        thread_recording.join()
                    print("Procesando audio...")
                    texto = transcribir_audio("comando.wav")
                    if texto:
                        comando = interpretar_comando_con_gemini(texto)
                        if comando:
                            print("DESDE MENU")
                            print(f"Comando interpretado: {comando}")
                        else:
                            print("❌ No se pudo interpretar el comando.")
                else:
                    print("⚠️ No hay grabación en curso para detener.")
             
            elif event.name == 'q':
                
                if recording:
                    stop_recording()
                    if thread_recording and thread_recording.is_alive():
                        thread_recording.join()
                print("👋 Saliendo del programa...")
                break
            else:
                print(f"🔍 Tecla '{event.name}' no reconocida. Usa 'r', 's' o 'q'."
                      )

if __name__ == "__main__":
    menu()