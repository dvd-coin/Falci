import os
from fastapi import FastAPI, UploadFile, File
from typing import List
import openai
import requests

app = FastAPI()

# Railway Variables kısmına eklediğin anahtarları buradan okuyacak
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@app.get("/")
def home():
    return {"message": "Kahve Falı API Çalışıyor!"}

@app.post("/predict-fortune")
async def predict_fortune(images: List[UploadFile] = File(...)):
    # 1. Fotoğrafları analiz için hazırla (Basitleştirilmiş mantık)
    # Gerçek uygulamada fotoğraflar base64'e çevrilip GPT-4o-mini'ye gönderilir.
    
    # AI'ya gönderilecek sistem talimatı
    prompt = "Sen bilge bir Türk kahvesi falcısısın. Gelen fincan görsellerini mistik, samimi ve detaylı yorumla."

    try:
        # OpenAI Chat Completion (Vision destekli model)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Lütfen bu fincanın falına bak."}
            ]
        )
        fortune_text = response.choices[0].message.content
        
        return {
            "fortune_text": fortune_text,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}
