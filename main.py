import os
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
        
        # Prompt specifico per ottenere la struttura che serve al tuo front
        prompt = """
        Analizza questa immagine (copertina di libro/film/gioco) e restituisci un JSON con questa struttura:
        {
            "tipo_contenuto": "film" o "libro" o "gioco",
            "dettagli": {
                "titolo": "Titolo originale",
                "eta_consigliata": "es. 6+",
                "riassunto": "Breve descrizione",
                "cover_url": null
            },
            "ratings": {
                "violenza": 0-5,
                "linguaggio": 0-5,
                "inclusivita": 0-5,
                "paura": 0-5
            },
            "alert_sicurezza": "Eventuali avvisi"
        }
        """

        # Chiamata con generazione JSON forzata
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=file.content_type)
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        # Converte la stringa JSON di Gemini in un dizionario Python
        import json
        analysis_result = json.loads(response.text)
        
        return analysis_result

    except Exception as e:
        print(f"Errore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Railway usa la porta 8080 di default
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)