import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from google.genai import types
import uvicorn

app = FastAPI()

# Configurazione Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        
        # --- MEGA PROMPT OTTIMIZZATO PER USARE LA FOTO SCATTATA ---
        prompt = """
        Sei un analista esperto di contenuti multimediali. Identifica l'opera nell'immagine.
        
        PROTOCOLLO:
        1. IDENTIFICAZIONE: Usa IMDb per confermare titolo e tipo (Videogioco, Film, Cartone, Libro).
        2. SICUREZZA: Usa Common Sense Media per i rating 0-5.
        3. FOTO: Non cercare URL di copertine esterne. Imposta sempre cover_url a null.

        RISPONDI SOLO IN JSON:
        {
            "tipo_contenuto": "string", 
            "dettagli": {
                "titolo": "TITOLO_IMDB",
                "eta_consigliata": "X+",
                "riassunto": "DESCRIZIONE_COMMON_SENSE",
                "cover_url": null
            },
            "ratings": {
                "violenza": 0-5,
                "linguaggio": 0-5,
                "inclusivita": 0-5,
                "paura": 0-5
            },
            "alert_sicurezza": "NOTE_DI_SICUREZZA"
        }
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=file.content_type)
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Errore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)