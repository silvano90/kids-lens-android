from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google import genai

import PIL.Image
import io
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializza il client con la nuova libreria
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        request_object_content = await file.read()
        img = PIL.Image.open(io.BytesIO(request_object_content))

        prompt = """
        Analizza l'immagine. Se è un cartone o film, identifica:
        1. Titolo
        2. Età consigliata (es: 3+, 6+, 12+, 18+)
        3. Breve riassunto (max 2 frasi)
        4. Alert sicurezza (violenza, linguaggio, etc.)
        
        Rispondi ESCLUSIVAMENTE con un oggetto JSON valido.
        """
        
        # Nuovo metodo di chiamata per google-genai
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, img]
        )
        
        # Pulizia e parsing del JSON
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
        
    except Exception as e:
        print(f"Errore: {e}")
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "KidsLens API (New SDK) is running!"}