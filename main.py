import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from google.genai import types
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        
        # PROMPT OTTIMIZZATO: Voti omogenei (Alto = Pericolo) + Sezione Episodi
        prompt = """
        IDENTIFICAZIONE: Analizza l'immagine (Cartoni/Serie TV/Film). Escludi Videogiochi.

        LOGICA RATING (Voto 0-5):
        IMPORTANTE: Per tutti i driver, un voto ALTO (5) indica un contenuto CRITICO/NEGATIVO, un voto BASSO (0) indica un contenuto SICURO.
        - violenza: 5 = molto violento.
        - paura: 5 = molto spaventoso.
        - linguaggio: 5 = linguaggio volgare/inappropriato.
        - carenza_inclusivita: 5 = presenza di stereotipi, pregiudizi o totale mancanza di diversità. 0 = estremamente inclusivo e positivo.

        RICERCA EPISODI: Cerca attivamente su IMDb e Common Sense Media se esistono episodi specifici segnalati dai genitori per scene disturbanti o contenuti horror nascosti.

        STRUTTURA JSON (SOLO JSON):
        {
            "tipo_contenuto": "cartone animato" | "film", 
            "dettagli": {
                "titolo": "TITOLO_UFFICIALE",
                "eta_consigliata": "X+",
                "riassunto": "Sintesi dell'opera",
                "cover_url": null
            },
            "ratings": {
                "violenza": {"voto": 0-5, "motivo": "..."},
                "paura": {"voto": 0-5, "motivo": "..."},
                "linguaggio": {"voto": 0-5, "motivo": "..."},
                "carenza_inclusivita": {"voto": 0-5, "motivo": "Spiega se ci sono stereotipi (voto alto) o se è inclusivo (voto basso)"}
            },
            "episodi_critici": [
                {"titolo": "Nome Episodio", "descrizione": "Spiega esattamente cosa succede di critico in questa puntata"}
            ],
            "alert_sicurezza": "Sintesi finale di attenzione per il genitore"
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
                temperature=0.0 
            )
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Errore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)