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
                Identifica ESCLUSIVAMENTE Cartoni, Serie TV o Film. IGNORA i videogiochi.
                
                PROTOCOLLO DI ANALISI:
                1. Identifica il titolo su IMDb.
                2. Analizza recensioni per genitori (tipo Common Sense Media) per trovare temi sensibili, stereotipi o scene disturbanti (es. horror nascosto in prodotti per bambini).
                3. RATING (0 a 5): 
                - violenza, paura, linguaggio: 5 = massimo pericolo/presenza.
                - carenza_inclusivita: 5 = presenza forte di stereotipi negativi, sessismo o mancanza di diversità. 0 = contenuto inclusivo e moderno.
                
                REGOLE JSON RIGIDE:
                - Non usare MAI virgolette doppie (") nei testi, usa solo virgolette singole (').
                - Non andare a capo nelle stringhe.
                - EPISODI_CRITICI: Trova 2-3 momenti reali (es. 'La morte di...', 'La scena del bullismo'). Se non sai il titolo, descrivi la scena. NON LASCIARE VUOTO.

                RISPONDI SOLO IN JSON:
                {
                    "tipo_contenuto": "cartone animato", 
                    "dettagli": {
                        "titolo": "...",
                        "eta_consigliata": "...",
                        "riassunto": "...",
                        "cover_url": null
                    },
                    "ratings": {
                        "violenza": {"voto": 0, "motivo": "Perché questo voto"},
                        "paura": {"voto": 0, "motivo": "Perché questo voto"},
                        "linguaggio": {"voto": 0, "motivo": "Perché questo voto"},
                        "carenza_inclusivita": {"voto": 0, "motivo": "Spiega quali stereotipi o perché è inclusivo"}
                    },
                    "episodi_critici": [
                        {"titolo": "Titolo o Scena", "descrizione": "Spiegazione dettagliata del pericolo"}
                    ],
                    "alert_sicurezza": "Sintesi dei rischi horror o comportamentali",
                    "spunti_conversazione": [
                        "Domanda per stimolare il pensiero critico",
                        "Domanda per gestire eventuali paure"
                    ]
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