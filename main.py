import os
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/predict-fortune")
async def predict_fortune(
    images: List[UploadFile] = File(...), 
    language: str = Form("tr")
):
    try:
        # Resimleri OpenAI'ın anlayacağı Base64 formatına çeviriyoruz
        image_messages = []
        for image in images:
            content = await image.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # OpenAI Vision çağrısı
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Vision desteği olan model
            messages=[
                {
                    "role": "system", 
                    "content": f"Sen mistik ve bilge bir Türk kahvesi falcısısın. Kullanıcının gönderdiği 3 adet fincan ve tabak görselini detaylıca incele. Gördüğün sembolleri (hayvanlar, eşyalar, yollar vb.) gerçek bir falcı gibi yorumla. Yanıtını sadece {language} dilinde ver."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Lütfen bu kahve falını benim için yorumlar mısın?"},
                        *image_messages # Resimler buraya ekleniyor
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return {"fortune_text": response.choices[0].message.content, "status": "success"}

    except Exception as e:
        print(f"Hata: {str(e)}")
        return {"fortune_text": f"Görsel analiz hatası: {str(e)}", "status": "error"}
