import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import google.generativeai as genai

from take_keys import takeKeys

KEY = takeKeys()  
genai.configure(api_key= KEY)

def grabar_audio(nombre_archivo="comando.wav", duracion=4, fs=44100):
    print("🎙️ Grabando tu voz...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(nombre_archivo, fs, audio)
    print("✅ Audio grabado.")

def transcribir_audio(nombre_archivo="comando.wav"):
    modelo = whisper.load_model("base")
    resultado = modelo.transcribe(nombre_archivo, language="es")
    texto = resultado["text"]
    print(f"📝 Texto reconocido: {texto}")
    return texto

def interpretar_comando_con_gemini(texto_usuario):
    prompt = f"""
Eres un asistente que interpreta comandos de voz para controlar un robot.
Convierte el siguiente mensaje en un JSON claro. Ejemplo:
Entrada: "ve a la caja azul" → Salida: {{"accion": "ir", "objeto": "caja azul"}}

Mensaje: "{texto_usuario}"
"""
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    respuesta = model.generate_content(prompt)
    print("🤖 Comando estructurado:")
    print(respuesta.text)
    return respuesta.text

# -------- PROGRAMA PRINCIPAL --------
grabar_audio()
texto = transcribir_audio()
comando = interpretar_comando_con_gemini(texto)
