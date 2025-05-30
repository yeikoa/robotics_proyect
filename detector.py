import google.generativeai as genai
from take_keys import takeKeys

KEY = takeKeys() 
genai.configure(api_key= KEY)

print("--- Iniciando búsqueda de modelos Gemini ---")
print("Modelos disponibles para el método 'generateContent' con tu clave:")
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"¡Ocurrió un error al intentar listar los modelos! Detalles: {e}")

print("--- Fin de la búsqueda de modelos Gemini ---")