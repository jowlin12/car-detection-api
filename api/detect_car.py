from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = Flask(__name__)

# Configuración de credenciales de Google Sheets
CREDENTIALS_PATH = "car-detection-api/credentials.json"  # Cambia esto por la ruta de tu archivo de credenciales
SHEET_NAME = "Base de Datos"  # Nombre de la hoja en el archivo

# Función para descargar y convertir imágenes a escala de grises
def download_image_as_grayscale(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")
        return img
    except Exception as e:
        raise RuntimeError(f"Error al descargar o procesar la imagen: {e}")


# Función para comparar imágenes usando ORB
def compare_images(img1, img2):
    try:
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        if descriptors1 is None or descriptors2 is None:
            raise ValueError("No se encontraron descriptores en una o ambas imágenes.")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        similarity = len(matches) / min(len(keypoints1), len(keypoints2))
        return similarity
    except Exception as e:
        raise RuntimeError(f"Error al comparar imágenes: {e}")


# Función para obtener datos de la hoja de Google Sheets
def get_references_from_sheet():
    try:
        # Autenticación con Google Sheets
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
        client = gspread.authorize(creds)

        # Abrir la hoja
        sheet = client.open_by_url(
            "https://docs.google.com/spreadsheets/d/1247WriRrUZSXep9Txj0398oXOtVWLnnI7JO5uS5pCGU/edit"
        ).worksheet(SHEET_NAME)

        # Leer datos de la hoja
        data = sheet.get_all_records()  # Lista de diccionarios con los datos de cada fila
        return data
    except Exception as e:
        raise RuntimeError(f"Error al obtener datos de Google Sheets: {e}")


@app.route('/api/detect', methods=['POST'])
def detect_car():
    try:
        # Validar datos de entrada
        data = request.get_json()
        if not data or "image_url" not in data:
            return jsonify({"error": "Debes proporcionar un campo 'image_url'."}), 400

        image_url = data["image_url"]

        # Descargar imagen de entrada
        input_image = download_image_as_grayscale(image_url)

        # Obtener referencias desde Google Sheets
        references = get_references_from_sheet()

        # Variables para la mejor coincidencia
        best_match = None
        highest_similarity = 0

        # Comparar la imagen con las referencias
        for ref in references:
            ref_image_url = ref["URL IMAGEN"]
            ref_brand = ref["MARCA"]
            ref_type = ref["TIPO"]

            ref_image = download_image_as_grayscale(ref_image_url)
            similarity = compare_images(input_image, ref_image)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = {"brand": ref_brand, "type": ref_type, "similarity": similarity}

        if best_match:
            return jsonify(best_match)
        else:
            return jsonify({"error": "No se encontró una coincidencia adecuada."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
