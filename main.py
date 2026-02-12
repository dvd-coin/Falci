import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from openai import OpenAI

app = FastAPI()

# CORS Ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI İstemcisi (API Key'i Railway Variables'dan çeker)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def read_root():
    return {"status": "online", "message": "Turkish Coffee Fortune API is ready!"}

@app.post("/predict-fortune")
async def predict_fortune(
    images: List[UploadFile] = File(...), 
    language: str = Form("tr")
):
    try:
        # OpenAI Çağrısı
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Hem hızlı hem ucuz
            messages=[
                {
                    "role": "system", 
                    "content": f"You are a mystical Turkish coffee fortune teller. Give a detailed, soulful, and traditional fortune reading in {language} language."
                },
                {
                    "role": "user", 
                    "content": "Interpret these coffee grounds images for my future."
                }
            ],
            max_tokens=500
        )
        
        fortune_text = response.choices[0].message.content
        return {"fortune_text": fortune_text, "status": "success"}

    except Exception as e:
        # Hata durumunda Base44'e ne olduğunu söyleyelim
        print(f"Hata oluştu: {str(e)}")
        return {"fortune_text": f"Bir hata oluştu: {str(e)}", "status": "error"}
