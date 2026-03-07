import os
import io
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google import genai # La libreria nuova!
from google.genai import types
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurazione Client Moderno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # 1. Preparazione immagine
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # 2. Configurazione generazione (Obblighiamo il JSON)
        # Usiamo il modello Flash che è velocissimo
        model_id = "gemini-2.5-flash" 
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "tipo_contenuto": {"type": "STRING"},
                    "dettagli": {
                        "type": "OBJECT",
                        "properties": {
                            "titolo": {"type": "STRING"},
                            "eta_consigliata": {"type": "STRING"},
                            "riassunto": {"type": "STRING"}
                        }
                    },
                    "alert_sicurezza": {"type": "STRING"}
                }
            }
        )

        print("--- INVIO RICHIESTA A GEMINI 2.0 ---")
        
        # 3. Chiamata
        response = client.models.generate_content(
            model=model_id,
            contents=["Analizza questa immagine per un'app di genitori e bambini.", img],
            config=config
        )

        # 4. Log per debug (Quello che volevi vedere tu)
        print("******************************************")
        print("RISPOSTA INTEGRALE GEMINI:")
        print(response.text)
        print("******************************************")

        return json.loads(response.text)

    except Exception as e:
        print(f"ERRORE CRITICO: {str(e)}")
        return {"error": str(e)}, 500

@app.get("/")
def health():
    return {"status": "online", "model": "gemini-2.5-flash"}