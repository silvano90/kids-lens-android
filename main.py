import os
import json
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content))

        # Configurazione con forzatura JSON
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config={"response_mime_type": "application/json"}
        )

        prompt = """
        Analizza l'immagine e restituisci un JSON con questa struttura:
        {
          "tipo_contenuto": "string",
          "dettagli": {
            "titolo": "string",
            "eta_consigliata": "string",
            "riassunto": "string"
          },
          "alert_sicurezza": "string"
        }
        """

        print("--- CHIAMATA A GEMINI IN CORSO ---")
        response = model.generate_content([prompt, img])
        
        # LOG CRUCIALE: Vedrai questo nei log di Railway
        print("******************************************")
        print("RISPOSTA INTEGRALE DI GEMINI IN CONSOLE:")
        print(response.text)
        print("******************************************")

        # Tentativo di parsing
        try:
            return json.loads(response.text)
        except Exception as parse_error:
            print(f"ERRORE PARSING JSON: {parse_error}")
            return {
                "error": "Gemini non ha risposto con un JSON valido",
                "raw_response": response.text
            }

    except Exception as e:
        print(f"ERRORE GENERALE: {str(e)}")
        return {"error": str(e)}, 500