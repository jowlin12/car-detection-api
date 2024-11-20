from flask import Flask, request, jsonify
import cv2
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
from io import BytesIO

app = Flask(__name__)

# Función para comparar imágenes
def compare_images(img1_url, img2_url):
    # Descargar imágenes
    img1 = cv2.imdecode(np.asarray(bytearray(requests.get(img1_url).content), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.asarray(bytearray(requests.get(img2_url).content), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    # Detectar y calcular características con ORB
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Comparar las características
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Calcular similitud
    similarity = len(matches) / min(len(keypoints1), len(keypoints2))
    return similarity

# Conexión a Google Sheets
def get_car_data(sheet_name, credentials_path):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name).sheet1
    data = sheet.get_all_records()
    return data

# Endpoint principal
@app.route('/api/detect', methods=['POST'])
def detect_car():
    try:
        # Obtener datos de la solicitud
        input_image_url = request.json.get("image_url")
        if not input_image_url:
            return jsonify({"error": "No image URL provided"}), 400

        # Cargar datos del sheet
        sheet_name = "CarDatabase"
        credentials_path = "credentials.json"  # Asegúrate de incluir esto en tu proyecto
        car_data = get_car_data(sheet_name, credentials_path)

        # Comparar con las imágenes de referencia
        best_match = None
        highest_similarity = 0

        for car in car_data:
            similarity = compare_images(input_image_url, car["Image URL"])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = car

        if best_match:
            return jsonify({"brand": best_match["Brand"], "type": best_match["Type"]})
        else:
            return jsonify({"error": "No matching car found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
