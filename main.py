from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

# Pegá tu token de Replicate aquí:
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/generar-imagen")

async def generar_imagen(request: Request):
    try:
        data = await request.json()
        replicate_token = data.get("replicate_token")

        if not replicate_token:
            return {"error": "No se recibió la API key"}

        # Crear cliente de Replicate con token recibido
        client = replicate.Client(api_token=replicate_token)

        # Aquí iría tu lógica para generar imagen
        # output = client.run(...)

        return {"status": "OK", "mensaje": "Token recibido y cliente creado"}

    except Exception as e:
        return {"error": f"Error inesperado: {e}"}

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
        data = await request.json()
        print("📥 Recibido del frontend:", data)

        prompt = data.get("prompt")
        if not prompt:
            return {"error": "Falta el prompt"}

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

        print(" Enviando a Replicate:", {
            "prompt": prompt,
            "image": image_url,
            "negative_prompt": negative_prompt,
        })
        
        print(" Respuesta de la aplicacion:", response.text)
        prediction = response.json()

        prediction_url = prediction.get("urls", {}).get("get")
        if not prediction_url:
            return {"error": "No se pudo obtener la URL de seguimiento del modelo"}

        # Esperar a que la imagen esté listas
        while True:
            result = requests.get(prediction_url, headers={"Authorization": f"Token {REPLICATE_TOKEN}"}).json()
            print(" Estado actual:", result["status"])

            if result["status"] == "succeeded":
             return {"imagen_generada": result["output"][0]}
            elif result["status"] == "failed":
             return {"error": "Fallo en la generación de imagen"}

    except Exception as e:
        print(" Error inesperado:", str(e))
        return {"error": f"Error en el backend: {str(e)}"}
    