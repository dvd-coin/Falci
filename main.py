import os
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from openai import OpenAI
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Google TTS Fonksiyonu
def get_google_tts_audio(text, lang_code):
    api_key = os.getenv("GOOGLE_API_KEY")
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    
    # Dil kodunu Google'ın anlayacağı formata çevir
    voice_lang = "tr-TR" if "Turk" in lang_code else "en-US"
    
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": voice_lang, "ssmlGender": "FEMALE"},
        "audioConfig": {"audioEncoding": "MP3"}
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["audioContent"]
    return None

@app.post("/predict-fortune")
async def predict_fortune(
    images: List[UploadFile] = File(...), 
    language: str = Form("Turkish")
):
    try:
        image_messages = []
        for image in images:
            content = await image.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # Karakteristik ve Kesin Dil Talimatı
        persona = (
            f"Sen dünya çapında meşhur, ağzı dualı, sezgileri çok kuvvetli bir falcı ablalar ablasısın. "
            f"ÖNEMLİ: Falı SADECE VE SADECE {language} dilinde anlatacaksın. Sakın başka dil kullanma! "
            f"Anlatımın samimi, sıcak olsun. 'Canım benim, bak şuraya' gibi cümleler kur. "
            f"Sanki karşında biri varmış gibi fala bak, analiz yapma, hayat hikayesi anlat."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": [*image_messages, {"type": "text", "text": "Hadi abla, dökül bakalım."}]}
            ],
            temperature=0.9
        )
        
        fortune_text = response.choices[0].message.content
        
        # Sesi Railway'de üret
        audio_content = get_google_tts_audio(fortune_text, language)
        
        return {
            "fortune_text": fortune_text, 
            "audio_base64": audio_content, # Ses verisini buraya ekledik
            "status": "success"
        }

    except Exception as e:
        return {"fortune_text": f"Gönül gözüm kapandı: {str(e)}", "status": "error"}
