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
        Analizza l'immagine (Cartone, Serie TV o Film). 
        REGOLE JSON RIGIDE: 
        1. Non usare MAI virgolette doppie (") nei testi, usa solo virgolette singole (').
        2. EPISODI CRITICI: Identifica 2-3 momenti specifici reali del contenuto che possono disturbare un bambino. Se non ricordi i titoli degli episodi, descrivi scene iconiche (es. 'La trasformazione del cattivo', 'La scena della tempesta'). NON LASCIARE VUOTO.
        3. RATING: 0-5. 'carenza_inclusivita' indica presenza di stereotipi (5 = molti stereotipi).

        STRUTTURA JSON:
        {
            "tipo_contenuto": "...", 
            "dettagli": {
                "titolo": "...",
                "eta_consigliata": "...",
                "riassunto": "...",
                "cover_url": null
            },
            "ratings": {
                "violenza": {"voto": 0, "motivo": "..."},
                "paura": {"voto": 0, "motivo": "..."},
                "linguaggio": {"voto": 0, "motivo": "..."},
                "carenza_inclusivita": {"voto": 0, "motivo": "..."}
            },
            "episodi_critici": [
                {"titolo": "Titolo Scena/Episodio", "descrizione": "Dettaglio del perché va attenzionato"}
            ],
            "alert_sicurezza": "...",
            "spunti_conversazione": ["...", "..."]
        }
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash-lite", 
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=file.content_type)
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2 
            )
        )

        try:
            return json.loads(response.text.strip())
        except json.JSONDecodeError:
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

    except Exception as e:
        print(f"Errore Backend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)