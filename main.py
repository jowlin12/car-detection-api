from flask import Flask, request, jsonify
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from PIL import Image
import requests
import numpy as np
import cv2

app = Flask(__name__)

# Configuración de Google Sheets
SPREADSHEET_ID = "1247WriRrUZSXep9Txj0398oXOtVWLnnI7JO5uS5pCGU"  # ID del Google Sheet
SHEET_NAME = "Base de Datos"  # Nombre de la hoja
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
GOOGLE_CREDENTIALS = "credentials.json"  # Ruta al archivo de credenciales

# Autenticación de Google Sheets
def get_sheet_data():
    credentials = Credentials.from_service_account_file(GOOGLE_CREDENTIALS, scopes=SCOPES)
    service = build("sheets", "v4", credentials=credentials)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=SHEET_NAME).execute()
    rows = result.get("values", [])
    return rows[1:]  # Excluye el encabezado

# Función para descargar imágenes desde URL y convertirlas a matrices
def download_image_as_array(url):
    try:
        response = requests.get(url)
        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Error descargando la imagen {url}: {e}")
        return None

# Función para comparar imágenes usando la diferencia estructural (SSIM)
def compare_images(img1, img2):
    try:
        img1 = cv2.cvtColor(cv2.resize(img1, (256, 256)), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(cv2.resize(img2, (256, 256)), cv2.COLOR_RGB2GRAY)
        score = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED).max()
        return score
    except Exception as e:
        print(f"Error comparando imágenes: {e}")
        return -1

@app.route("/compare", methods=["POST"])
def compare():
    # Obtener URL de la imagen enviada
    data = request.get_json()
    if "image_url" not in data:
        return jsonify({"error": "Falta 'image_url' en el cuerpo de la solicitud"}), 400

    input_url = data["image_url"]
    input_image = download_image_as_array(input_url)
    if input_image is None:
        return jsonify({"error": "No se pudo descargar la imagen proporcionada"}), 400

    # Obtener imágenes y datos del Google Sheet
    rows = get_sheet_data()
    if not rows:
        return jsonify({"error": "No se encontraron datos en la hoja"}), 500

    best_match = {"brand": None, "type": None, "score": -1}

    # Comparar la imagen con todas las imágenes en el Sheet
    for row in rows:
        image_url, brand, car_type = row
        sheet_image = download_image_as_array(image_url)
        if sheet_image is None:
            continue

        score = compare_images(input_image, sheet_image)
        if score > best_match["score"]:
            best_match = {"brand": brand, "type": car_type, "score": score}

    if best_match["score"] == -1:
        return jsonify({"error": "No se encontraron coincidencias"}), 404

    return jsonify({
        "brand": best_match["brand"],
        "type": best_match["type"],
        "similarity_score": best_match["score"]
    })

if __name__ == "__main__":
    app.run(debug=True)
