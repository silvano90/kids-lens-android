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
        IDENTIFICAZIONE OBBLIGATORIA: Analizza l'immagine e identifica esclusivamente Cartoni Animati, Serie TV o Film. 
        IGNORA CATEGORICAMENTE l'opzione 'Videogioco'. Se l'opera esiste in più formati, analizza la versione ANIMATA.

        PROCESSO DI VERIFICA (IMDb & Common Sense Media):
        1. Identifica il titolo esatto.
        2. Cerca attivamente le 'User Reviews' su IMDb e la sezione 'What Parents Need to Know' su Common Sense Media.
        3. CASI CRITICI: Cerca specificamente menzioni di episodi speciali, scene disturbanti o contenuti 'horror' (es. Bread Barbershop ha episodi con atmosfere horror che non compaiono nella descrizione standard).
        4. Se trovi segnalazioni di genitori su scene di paura, il rating 'paura' deve riflettere il picco massimo dell'opera, non la media.

        STRUTTURA JSON (NON AGGIUNGERE ALTRO TESTO):
        {
            "tipo_contenuto": "cartone animato" | "film", 
            "dettagli": {
                "titolo": "TITOLO_UFFICIALE_IMDB",
                "eta_consigliata": "X+",
                "riassunto": "Sintesi che includa avvertimenti sulle puntate particolari",
                "cover_url": null
            },
            "ratings": {
                "violenza": 0-5,
                "linguaggio": 0-5,
                "inclusivita": 0-5,
                "paura": 0-5
            },
            "alert_sicurezza": "DETTAGLIA QUI LE PUNTATE O LE SCENE SPECIFICHE SEGNALATE DAI GENITORI (es. scene horror o tensione)"
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