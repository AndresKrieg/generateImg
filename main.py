from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from fastapi.responses import JSONResponse

# Pegá tu token de Replicate aquí:
REPLICATE_TOKEN = os.getenv("REPLICATE_TOKEN")

# ✅ Versión válida del modelo estable gratuito
MODEL_VERSION = "95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"

app = FastAPI()

# Configurar CORS para permitir llamadas desde el frontend
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
        # Obtener los datos de la solicitud
        data = await request.json()
        print("📥 Recibido del frontend:", data)

        # Verificar que el 'prompt' esté presente
        prompt = data.get("prompt")
        if not prompt:
            return JSONResponse(content={"error": "Falta el prompt"}, status_code=400)

        # URLs de las imágenes de fondo y máscara (asegurarse de que estas sean correctas)
        image_url = "https://www.incolmotos-yamaha.com.co/wp-content/uploads/2025/04/yamahaNmaxBgImgAI.jpg"
        image_mask = "https://www.incolmotos-yamaha.com.co/wp-content/uploads/2025/04/yamahaNmaxBgImgAI-mask.jpg"
        negative_prompt = "blurry, two riders, respect the mask, distorted, extra limbs, modify mask, floating objects, surreal background, unrealistic lighting, low quality, wrong colors, vehicle flying, deformed rider, shadows missing, duplicated wheels, glitch, abstract art\n"

        # Petición al modelo gratuito (texto a imagen)
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

        # Verificar la respuesta de la petición al modelo
        if response.status_code != 200:
            return JSONResponse(content={"error": "Error al generar la imagen en Replicate"}, status_code=500)

        print(" Enviando a Replicate:", {
            "prompt": prompt,
            "image": image_url,
            "negative_prompt": negative_prompt,
        })

        # Obtener la URL de la imagen generada
        prediction = response.json()
        prediction_url = prediction.get("urls", {}).get("get")

        if not prediction_url:
            return JSONResponse(content={"error": "No se pudo obtener la URL de seguimiento del modelo"}, status_code=500)

        # Esperar a que la imagen esté lista
        while True:
            result = requests.get(prediction_url, headers={"Authorization": f"Token {REPLICATE_TOKEN}"}).json()
            print(" Estado actual:", result["status"])

            if result["status"] == "succeeded":
                # Devolver la imagen generada
                return JSONResponse(content={"imagen_generada": result["output"][0]}, status_code=200)
            elif result["status"] == "failed":
                return JSONResponse(content={"error": "Fallo en la generación de imagen"}, status_code=500)

    except Exception as e:
        print(" Error inesperado:", str(e))
        return JSONResponse(content={"error": f"Error en el backend: {str(e)}"}, status_code=500)

