from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

# Obtiene el token desde las variables de entorno
REPLICATE_TOKEN = os.getenv("REPLICATE_TOKEN")

# Versión del modelo gratuito de Replicate
MODEL_VERSION = "95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3" 

app = FastAPI()

# CORS para permitir conexión desde cualquier origen (puedes limitarlo si quieres)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/generar-imagen")
async def generar_imagen(request: Request):
    try:
        headers = request.headers
        print("🔐 Authorization Header:", headers.get("authorization"))

        data = await request.json()
        print("📥 Recibido del frontend:", data)

        prompt = data.get("prompt")
        if not prompt:
            return {"error": "Falta el prompt"}

        # Imagen base y máscara fija
        image_url = "https://www.incolmotos-yamaha.com.co/wp-content/uploads/2025/04/yamahaNmaxBgImgAI.jpg"
        image_mask = "https://www.incolmotos-yamaha.com.co/wp-content/uploads/2025/04/yamahaNmaxBgImgAI-mask.jpg"

        negative_prompt = (
            "blurry, two riders, respect the mask, distorted, extra limbs, modify mask, "
            "floating objects, surreal background, unrealistic lighting, low quality, wrong colors, "
            "vehicle flying, deformed rider, shadows missing, duplicated wheels, glitch, abstract art"
        )

        # Enviar solicitud a Replicate
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {REPLICATE_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "version": MODEL_VERSION,
                "input": {
                    "hdr": 0,
                    "mask": image_mask,
                    "image": image_url,
                    "steps": 25,
                    "width": 1024,
                    "height": 512,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "scheduler": "DPMSolverMultistep",
                    "creativity": 0.2,
                    "resolution": "original",
                    "resemblance": 0.5,
                    "guidance_scale": 7.5
                }
            }
        )

        print("📤 Enviado a Replicate:", prompt)
        prediction = response.json()

        prediction_url = prediction.get("urls", {}).get("get")
        if not prediction_url:
            return {"error": "No se pudo obtener la URL de seguimiento del modelo"}

        # Esperar hasta que la predicción esté lista
        while True:
            result = requests.get(
                prediction_url,
                headers={"Authorization": f"Token {REPLICATE_TOKEN}"}
            ).json()

            print("⏳ Estado actual:", result["status"])

            if result["status"] == "succeeded":
                return {"imagen_generada": result["output"][0]}
            elif result["status"] == "failed":
                return {"error": "Fallo en la generación de imagen"}

    except Exception as e:
        print("❌ Error inesperado:", str(e))
        return {"error": f"Error en el backend: {str(e)}"}
