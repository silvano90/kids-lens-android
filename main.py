import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from google.genai import types
import uvicorn

app = FastAPI()

# Configurazione Client con la nuova libreria
# Assicurati che la variabile d'ambiente GEMINI_API_KEY sia impostata su Railway
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

@app.get("/")
def read_root():
    return {"status": "online", "model": "gemini-2.5-flash-lite"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        
        # --- IL MEGA PROMPT RECUPERATO E POTENZIATO ---
        prompt = """
        Sei un analista esperto di contenuti multimediali per famiglie. 
        Il tuo compito è identificare con precisione assoluta l'opera nell'immagine.

        ### PROTOCOLLO DI ANALISI:
        1. IDENTIFICAZIONE: Cerca il titolo esatto su IMDb. Distingui categoricamente tra Videogioco, Film, Serie TV (Cartone) o Libro. 
           - Se vedi interfacce (HP, tasti, menu), è un VIDEOGIOCO.
           - Se vedi uno stile animato senza elementi di gioco, è un CARTONE/FILM.
        2. RICERCA SICUREZZA: Consulta i parametri di Common Sense Media per l'opera identificata.
        3. RATING (Scala 0-5):
           - Violenza: Presenza di scontri, sangue o aggressività.
           - Linguaggio: Parole scurrili o concetti inappropriati.
           - Inclusività: Rappresentazione di diversità e messaggi positivi.
           - Paura: Scene buie, mostri o tensione psicologica.

        ### REGOLE DI RISPOSTA:
        - Restituisci ESCLUSIVAMENTE un JSON.
        - Sii oggettivo e severo sui rating.
        - Se non sei sicuro, indica il contenuto più probabile basandoti sui dati IMDb.

        ### STRUTTURA JSON RICHIESTA:
        {
            "tipo_contenuto": "cartone animato", 
            "dettagli": {
                "titolo": "TITOLO_UFFICIALE_IMDB",
                "eta_consigliata": "X+_BASATA_SU_COMMON_SENSE",
                "riassunto": "DESCRIZIONE_BASATA_SU_COMMON_SENSE",
                "cover_url": "URL_LOCANDINA_SE_DISPONIBILE"
            },
            "ratings": {
                "violenza": 0,
                "linguaggio": 0,
                "inclusivita": 0,
                "paura": 0
            },
            "alert_sicurezza": "DETTAGLI_SUI_RISCHI_SPECIFICI"
        }
        """

        # Esecuzione con Gemini 2.5 Flash Lite
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=file.content_type)
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2 # Più basso per essere meno "creativo" e più preciso
            )
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Errore nel server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Railway usa la porta 8080 di default
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)