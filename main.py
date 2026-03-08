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
        # Lettura del file caricato
        image_data = await file.read()
        
        # Prompt per l'analisi (personalizzalo come preferisci)
        prompt = "Analizza questa immagine in dettaglio e descrivi cosa vedi."

        # Chiamata al modello 2.5 Flash Lite
        # Nota: Se 'gemini-2.5-flash-lite' desse 404, prova 'gemini-2.0-flash-lite'
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_data,
                    mime_type=file.content_type
                )
            ]
        )

        return {
            "filename": file.filename,
            "analysis": response.text
        }

    except Exception as e:
        print(f"Errore durante l'analisi: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Railway usa la porta 8080 di default
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)