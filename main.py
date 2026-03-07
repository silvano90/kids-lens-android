import os
import json
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import re

# Configurazione API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# Permette all'app React Native di comunicare con il server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT_KIDS = """
Analizza questa immagine (probabilmente un cartone animato o un giocattolo). 
Rispondi ESCLUSIVAMENTE con un oggetto JSON in questo formato:
{
  "tipo_contenuto": "film/serie/giocattolo",
  "dettagli": {
    "titolo": "Nome originale",
    "eta_consigliata": "fascia d'età (es. 3+, 6+)",
    "riassunto": "Breve descrizione semplice per genitori"
  },
  "alert_sicurezza": "Eventuali temi sensibili o conferma di contenuti sicuri"
}
Non aggiungere testo prima o dopo il JSON.
"""

def try_gemini_analysis(img):
    model = genai.GenerativeModel('gemini-1.5-flash')
    # Forziamo il modello a essere sintetico
    response = model.generate_content([PROMPT_KIDS, img])
    
    print(f"DEBUG - Risposta grezza Gemini: {response.text}") # Log su Railway
    
    # Cerchiamo il pattern JSON { ... } nel testo usando Regex
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json.loads(json_str)
    
    # Se la regex fallisce, proviamo il metodo standard
    clean_text = response.text.replace('```json', '').replace('```', '').strip()
    return json.loads(clean_text)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Legge l'immagine inviata dal telefono
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content))

        max_retries = 2
        last_exception = None

        for attempt in range(max_retries):
            try:
                # Tenta l'analisi
                analysis_data = try_gemini_analysis(img)
                print(f"Tentativo {attempt + 1}: Successo!")
                return analysis_data
            except json.JSONDecodeError as je:
                last_exception = je
                print(f"Tentativo {attempt + 1}: JSON non valido, riprovo...")
                continue
            except Exception as e:
                # Se l'errore è bloccante (Quota, API Key, etc.), usciamo subito
                error_msg = str(e).lower()
                if "quota" in error_msg or "key" in error_msg or "429" in error_msg:
                    return {"error": f"Errore critico API: {str(e)}"}, 500
                last_exception = e
                continue

        # Se arriviamo qui, i tentativi sono falliti
        return {
            "tipo_contenuto": "errore_analisi",
            "dettagli": {
                "titolo": "Errore di decodifica",
                "eta_consigliata": "?",
                "riassunto": "Gemini non ha risposto in modo strutturato dopo due tentativi."
            },
            "alert_sicurezza": "Nessuna informazione disponibile."
        }

    except Exception as e:
        print(f"Errore generale: {str(e)}")
        return {"error": f"Errore del server: {str(e)}"}, 500

@app.get("/")
def home():
    return {"status": "KidsLens Backend Online"}