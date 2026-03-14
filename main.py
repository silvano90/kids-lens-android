import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from google.genai import types
import uvicorn

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        
        # --- PROMPT BLINDATO: NO VIDEOGIOCHI + DEEP SEARCH ---
prompt = """
        Analizza l'opera identificata (Solo Film/Cartoni).
        
        PROTOCOLLO DI ANALISI DETTAGLIATA:
        1. CATEGORIE: Per Violenza, Linguaggio, Inclusività e Paura, identifica il motivo esatto del voto basandoti su IMDb Parental Guide e Common Sense Media.
        2. EPISODI CRITICI: Trova i titoli degli episodi o le scene specifiche che hanno causato i rating più alti (es. la puntata horror di Bread Barbershop).
        3. SPUNTI: Crea domande educative.

        RISPONDI SOLO IN JSON CON QUESTA STRUTTURA:
        {
            "tipo_contenuto": "cartone animato",
            "dettagli": { "titolo": "...", "eta_consigliata": "...", "riassunto": "..." },
            "ratings": {
                "violenza": {"voto": 0, "motivo": "..."},
                "linguaggio": {"voto": 0, "motivo": "..."},
                "inclusivita": {"voto": 0, "motivo": "..."},
                "paura": {"voto": 0, "motivo": "..."}
            },
            "episodi_critici": [
                {"titolo": "...", "descrizione": "..."}
            ],
            "alert_sicurezza": "...",
            "spunti_conversazione": ["...", "..."]
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
                temperature=0.0 # Forza la massima precisione e zero creatività
            )
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Errore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)