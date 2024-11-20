from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests

app = Flask(__name__)


# Función para descargar y convertir imágenes a escala de grises
def download_image_as_grayscale(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Verifica errores HTTP
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")
        return img
    except Exception as e:
        raise RuntimeError(f"Error al descargar o procesar la imagen: {e}")


# Función para comparar dos imágenes usando características ORB
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


@app.route('/api/detect', methods=['POST'])
def detect_car():
    try:
        # Validar datos de entrada
        data = request.get_json()
        if not data or "image_url" not in data or "references" not in data:
            return jsonify({"error": "El cuerpo debe incluir 'image_url' y 'references'."}), 400

        image_url = data["image_url"]
        references = data["references"]  # Lista de referencias { "brand": ..., "type": ..., "image_url": ... }

        if not isinstance(references, list) or len(references) == 0:
            return jsonify({"error": "El campo 'references' debe ser una lista con datos."}), 400

        # Descargar imagen de entrada
        input_image = download_image_as_grayscale(image_url)

        # Variables para almacenar la mejor coincidencia
        best_match = None
        highest_similarity = 0

        # Comparar la imagen de entrada con las referencias
        for ref in references:
            if "brand" not in ref or "type" not in ref or "image_url" not in ref:
                continue  # Ignorar referencias incompletas

            ref_image_url = ref["image_url"]
            ref_image = download_image_as_grayscale(ref_image_url)
            similarity = compare_images(input_image, ref_image)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = {"brand": ref["brand"], "type": ref["type"], "similarity": similarity}

        if best_match:
            return jsonify(best_match)
        else:
            return jsonify({"error": "No se encontró una coincidencia adecuada."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
