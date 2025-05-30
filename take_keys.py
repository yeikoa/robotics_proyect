from dotenv import load_dotenv
import os

def takeKeys():
    load_dotenv(".env.local")
    KEY_GEMINI = os.getenv("KEY_GEMINI")
    if not KEY_GEMINI:
        raise ValueError("Por favor, define la variable de entorno")
    return KEY_GEMINI

