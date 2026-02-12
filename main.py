import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from openai import OpenAI

app = FastAPI()

# CORS Ayarları - Base44 bağlantısı için hayati önemde
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI İstemcisi
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def home():
    return {"status": "online", "project": "Turkish Coffee Fortune"}

@app.post("/predict-fortune")
async def predict_fortune(
    images: List[UploadFile] = File(...), 
    language: str = Form("en")
):
    try:
        # GPT-4o-mini kullanarak analiz yapıyoruz
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a professional Turkish coffee fortune teller. Provide a mystical and detailed reading in {language} language."
                },
                {
                    "role": "user", 
                    "content": "Interpret the coffee grounds in these images."
                }
            ],
            max_tokens=1000
        )
        
        # Base44'ün beklediği anahtar: fortune_text
        result = response.choices[0].message.content
        return {
            "fortune_text": result,
            "status": "success"
        }

    except Exception as e:
        print(f"Hata: {str(e)}")
        return {
            "fortune_text": f"Bir hata oluştu: {str(e)}",
            "status": "error"
        }
