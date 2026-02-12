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
        image_messages = []
        for image in images:
            content = await image.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # Talimatı çok daha net ve zorunlu hale getirdik
        response = client.chat.completions.create(
            model="gpt-4o", # Modeli GPT-4o (Görsel uzmanı) olarak güncelledik
            messages=[
                {
                    "role": "system", 
                    "content": (
                        f"Sen profesyonel bir falcısın. Sana gönderilen resimler gerçek kahve fincanı fotoğraflarıdır. "
                        f"ASLA 'ben yapay zekayım' veya 'genel yorum yapabilirim' deme. "
                        f"Doğrudan resimlerde gördüğün şekilleri (örneğin: 'sağ tarafta bir at görüyorum', 'fincanın dibinde bir yol var') anlatarak başla. "
                        f"Mistik, geleneksel ve gizemli bir dil kullan. Yanıtın sadece {language} dilinde olsun."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Bu fincan benim, lütfen içindeki şekilleri yorumla ve geleceğimi söyle."},
                        *image_messages 
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.7 # Biraz daha yaratıcı ve falcı gibi konuşması için
        )
        
        return {"fortune_text": response.choices[0].message.content, "status": "success"}

    except Exception as e:
        return {"fortune_text": f"Görsel analiz hatası: {str(e)}", "status": "error"}
