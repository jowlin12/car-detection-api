from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
import io
from PIL import Image

# Configuración de Google Sheets
SPREADSHEET_ID = "1247WriRrUZSXep9Txj0398oXOtVWLnnI7JO5uS5pCGU"
SHEET_NAME = "Base de Datos"
CREDENTIALS_PATH = "credentials.json"  # Ruta al archivo de credenciales

# Configuración de Flask
app = Flask(__name__)

# Función para descargar imágenes desde una URL
def descargar_imagen(url):
    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error al descargar la imagen: {e}")
        return None

# Función para obtener datos del Google Sheet
def obtener_datos_sheet():
    try:
        credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        service = build('sheets', 'v4', credentials=credentials)
        sheet = service.spreadsheets()

        # Leer datos del sheet
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=SHEET_NAME).execute()
        rows = result.get('values', [])
        return rows[1:]  # Excluye la fila de encabezados
    except HttpError as error:
        print(f"Error al acceder a Google Sheets: {error}")
        return []

# Función para comparar imágenes usando histogramas
def calcular_similitud(imagen1, imagen2):
    try:
        hist1 = cv2.calcHist([imagen1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([imagen2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    except Exception as e:
        print(f"Error al calcular similitud: {e}")
        return 0

@app.route('/api/detect', methods=['POST'])
def detectar_imagen():
    try:
        data = request.get_json()
        url_imagen = data.get('image_url')

        if not url_imagen:
            return jsonify({"error": "Falta la URL de la imagen"}), 400

        # Descargar imagen a clasificar
        imagen_principal = descargar_imagen(url_imagen)
        if imagen_principal is None:
            return jsonify({"error": "No se pudo descargar la imagen principal"}), 400

        # Obtener datos del Google Sheet
        datos_sheet = obtener_datos_sheet()

        # Variables para rastrear la mejor coincidencia
        mejor_similitud = 0
        mejor_resultado = {"marca": None, "tipo": None}

        # Comparar con imágenes del sheet
        for fila in datos_sheet:
            url_base, marca, tipo = fila
            imagen_base = descargar_imagen(url_base)
            if imagen_base is None:
                continue

            similitud = calcular_similitud(imagen_principal, imagen_base)
            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor_resultado = {"marca": marca, "tipo": tipo}

        return jsonify(mejor_resultado)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
