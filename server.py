from flask import Flask, request, jsonify

app = Flask(__name__)
objetos_actuales = []

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "running"}), 200


@app.route("/comando", methods=["POST"])
def recibir_comando():
    global objetos_actuales
    data = request.get_json()
    print("Comando recibido:", data)

    if "objeto" in data:
        objetos_actuales = [data["objeto"]]
    elif "objetos" in data:
        objetos_actuales = data["objetos"]
    return jsonify({"respuesta": "comando recibido"}), 200

@app.route("/objetivos", methods=["GET"])
def obtener_objetivos():
    print("Objetivos actuales:", objetos_actuales)
    return jsonify({"objetos": objetos_actuales}), 200
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
