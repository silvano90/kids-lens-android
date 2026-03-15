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
        
        prompt = """
        Analizza l'immagine e identifica Cartoni Animati, Serie TV o Film. 
        REGOLE JSON RIGIDE: 
        1. Non usare MAI virgolette doppie (") dentro i testi (es. riassunto o motivi), usa solo virgolette singole (').
        2. Non andare mai a capo (\n) dentro le stringhe.
        3. Rating 0-5: Alto = Pericolo/Negativo. 'carenza_inclusivita' 5 significa molti stereotipi, 0 significa perfetto.

        STRUTTURA JSON:
        {
            "tipo_contenuto": "cartone animato", 
            "dettagli": {
                "titolo": "TITOLO",
                "eta_consigliata": "X+",
                "riassunto": "Sintesi senza virgolette doppie",
                "cover_url": null
            },
            "ratings": {
                "violenza": {"voto": 0, "motivo": "testo"},
                "paura": {"voto": 0, "motivo": "testo"},
                "linguaggio": {"voto": 0, "motivo": "testo"},
                "carenza_inclusivita": {"voto": 0, "motivo": "testo"}
            },
            "episodi_critici": [
                {"titolo": "Episodio X", "descrizione": "Perché è critico"}
            ],
            "alert_sicurezza": "Sintesi finale",
            "spunti_conversazione": ["domanda 1", "domanda 2"]
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

        # Pulizia e Parsing
        try:
            return json.loads(response.text.strip())
        except json.JSONDecodeError:
            # Fallback se l'AI mette i backticks ```json
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

    except Exception as e:
        print(f"Errore: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)