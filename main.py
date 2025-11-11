import os
import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

# === 🚀 Crear app FastAPI ===
app = FastAPI()

# === ⚙️ CORS (ajusta dominios permitidos si quieres restringir) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 📦 Cargar modelos ===
BASE_DIR = os.path.dirname(__file__)
print("Cargando modelos...")

model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "modelo_plantas.h5"))
feature_extractor = tf.keras.models.load_model(os.path.join(BASE_DIR, "models", "feature_extractor.h5"))
pca = joblib.load(os.path.join(BASE_DIR, "models", "pca_flores.pkl"))
kmeans = joblib.load(os.path.join(BASE_DIR, "models", "kmeans_flores.pkl"))
cluster_to_class = np.load(os.path.join(BASE_DIR, "models", "cluster_to_class.npy"), allow_pickle=True).item()
class_names = np.load(os.path.join(BASE_DIR, "models", "class_names.npy"))

print("✅ Modelos cargados correctamente\n")

# === 🧠 Funciones auxiliares ===
def preprocesar_imagen_pil(pil_img):
    """Preprocesa una imagen PIL para MobileNetV2."""
    img = pil_img.resize((180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def predecir_clase(pil_img):
    """Clasificación supervisada (modelo softmax)."""
    img_array = preprocesar_imagen_pil(pil_img)
    preds = model.predict(img_array, verbose=0)
    clase = class_names[np.argmax(preds)]
    confianza = float(np.max(preds))
    return clase, confianza

def asignar_cluster(pil_img):
    """Agrupamiento no supervisado con KMeans + PCA."""
    img_array = preprocesar_imagen_pil(pil_img)
    features = feature_extractor.predict(img_array, verbose=0)
    features_pca = pca.transform(features)
    cluster_id = kmeans.predict(features_pca)[0]
    mapped_class = class_names[cluster_to_class[cluster_id]]
    return int(cluster_id), str(mapped_class)

# === 🧾 Endpoint principal ===
@app.post("/predict")
async def predict_flower(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_img = Image.open(BytesIO(contents)).convert("RGB")

        clase_pred, conf = predecir_clase(pil_img)
        cluster_id, clase_cluster = asignar_cluster(pil_img)

        return JSONResponse(
            content={
                "success": True,
                "class_supervised": str(clase_pred),
                "confidence": round(conf, 3),
                "cluster_id": cluster_id,
                "class_cluster": str(clase_cluster)
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)
