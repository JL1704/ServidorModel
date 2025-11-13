import requests

# URL del servidor en Render
url = "https://servidormodel.onrender.com/predict"

# Ruta local a tu imagen
image_path = r"C:\Users\josel\Repositorios\datasetFlores\pruebas\rosa.jpg"

# Enviar la imagen como form-data
with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

# Mostrar respuesta
print("Código de estado:", response.status_code)
print("Respuesta del servidor:")
print(response.json())

