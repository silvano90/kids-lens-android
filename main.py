import os
import io
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # SCHEMA RIGIDO: Forza Gemini a rispondere con numeri e stringhe pulite
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "tipo_contenuto": {"type": "STRING"},
                "dettagli": {
                    "type": "OBJECT",
                    "properties": {
                        "titolo": {"type": "STRING"},
                        "eta_consigliata": {"type": "STRING", "description": "Formato fisso: '3+', '6+', '12+', '18+'"},
                        "riassunto": {"type": "STRING"},
                        "cover_url": {"type": "STRING", "description": "URL diretto dell'immagine ufficiale del titolo (locandina)"}
                    }
                },
                "ratings": {
                    "type": "OBJECT",
                    "properties": {
                        "violenza": {"type": "INTEGER", "description": "0-5"},
                        "linguaggio": {"type": "INTEGER", "description": "0-5"},
                        "inclusivita": {"type": "INTEGER", "description": "0-5"},
                        "paura": {"type": "INTEGER", "description": "0-5"}
                    }
                },
                "alert_sicurezza": {"type": "STRING"}
            },
            "required": ["tipo_contenuto", "dettagli", "ratings"]
        }

        # PROMPT STRATEGICO
        prompt = """
        Sei un esperto di media per bambini. Identifica il contenuto nell'immagine.
        1. Consulta mentalmente database come Common Sense Media o IMDB.
        2. Restituisci l'età consigliata SOLO come una delle seguenti etichette: '3+', '6+', '12+', '18+'.
        3. Per 'cover_url', trova l'URL statico della locandina o della cover ufficiale.
        4. Valuta da 0 a 5 i driver di sicurezza (violenza, linguaggio, inclusivita, paura).
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2 # Più basso per risposte meno creative e più precise
            )
        )

        print("--- RISPOSTA GEMINI ANALYTICS ---")
        print(response.text)
        
        return json.loads(response.text)

    except Exception as e:
        print(f"ERRORE: {str(e)}")
        return {"error": str(e)}, 500