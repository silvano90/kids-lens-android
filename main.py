import os
import json
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, UploadFile
from google import generativeai as genai

app = FastAPI()

# Configurazione Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# Funzione per lo Scraping della Cover
def get_imdb_cover(imdb_id):
    if not imdb_id or imdb_id == "n/a":
        return None
    try:
        url = f"https://www.imdb.com/title/{imdb_id}/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        image_tag = soup.find("meta", property="og:image")
        return image_tag["content"] if image_tag else None
    except:
        return None

@app.post("/analyze")
async def analyze_cartone(file: UploadFile = File(...)):
    # Leggiamo l'immagine
    image_data = await file.read()
    
    # --- STEP 1: IDENTIFICAZIONE ---
    # Chiediamo a Gemini solo il titolo per essere veloci
    id_prompt = "Identifica il cartone animato o film in questa immagine. Restituisci solo il titolo e l'anno."
    id_response = model.generate_content([id_prompt, {"mime_type": "image/jpeg", "data": image_data}])
    titolo_rilevato = id_response.text.strip()
    
    # --- STEP 2: ANALISI STRUTTURATA (Il Super-Prompt Blindato) ---
    analysis_prompt = f"""
    Analizza il contenuto: {titolo_rilevato}.
    Usa come fonti Common Sense Media e IMDb.
    Rispondi ESCLUSIVAMENTE in formato JSON con questa struttura:
    {{
      "id_imdb": "ID (es. tt1234567)",
      "titolo_ufficiale": "Titolo",
      "target_eta": "Età consigliata",
      "rating_generale": 0-100,
      "pilastri": {{
        "paura": {{"livello": "Basso/Medio/Alto", "perche": "..."}},
        "lutto": {{"livello": "Basso/Medio/Alto", "perche": "..."}},
        "capitalismo": {{"livello": "Basso/Medio/Alto", "perche": "..."}},
        "bullismo": {{"livello": "Basso/Medio/Alto", "perche": "..."}},
        "linguaggio": {{"livello": "Basso/Medio/Alto", "perche": "..."}},
        "sostanze": {{"livello": "Basso/Medio/Alto", "perche": "..."}}
      }},
      "fonti": {{"csm": "riassunto esperti", "imdb": "riassunto genitori"}},
      "verdetto_finale": "Breve sintesi",
      "consiglio_pratico": "Domanda per il bambino"
    }}
    Nota: Sii equilibrato. Non penalizzare contenuti educativi profondi.
    """
    
    analysis_response = model.generate_content(analysis_prompt)
    
    # Pulizia della risposta (a volte Gemini mette i backticks ```json)
    json_text = analysis_response.text.replace("```json", "").replace("```", "").strip()
    result = json.loads(json_text)
    
    # --- STEP 3: SCRAPING COVER ---
    cover_url = get_imdb_cover(result.get("id_imdb"))
    result["cover_url"] = cover_url if cover_url else "[https://via.placeholder.com/300x450?text=No+Cover](https://via.placeholder.com/300x450?text=No+Cover)"

    return result