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
    language: str = Form(...) 
):
    # 1. LOGLAMA: Gelen dili kesin görelim
    print(f"--- GELEN DİL İSTEĞİ: {language} ---")

    # 2. DİL AYARI (Basit ve Sağlam)
    # Eğer içinde 'en' veya 'Eng' geçiyorsa İngilizce yap, yoksa Türkçe kal.
    if language and "en" in language.lower():
        selected_lang = "English"
        system_lang_instruction = "Give the response ONLY in English."
    else:
        selected_lang = "Turkish"
        system_lang_instruction = "Yanıtı SADECE Türkçe ver."

    try:
        image_messages = []
        for image in images:
            content = await image.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # 3. FALCI (Kısa ve Öz - Kotan gitmesin diye token kıstım)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        f"Sen mistik bir falcısın. {system_lang_instruction} "
                        "Falı 3 kısa paragraf halinde, gizemli bir tonla anlat."
                    )
                },
                {"role": "user", "content": [*image_messages, {"type": "text", "text": "Falımı yorumla."}]}
            ],
            max_tokens=600 # Kotan için azalttım
        )
        
        fortune_text = response.choices[0].message.content
        
        # 4. SES (KESİN MP3 FORMATI)
        audio_response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=fortune_text,
            response_format="mp3" # Formatı zorluyoruz
        )
        
        audio_base64 = base64.b64encode(audio_response.content).decode('utf-8')
        
        return {
            "fortune_text": fortune_text, 
            "audio_base64": audio_base64,
            "detected_language": selected_lang,
            "status": "success"
        }

    except Exception as e:
        print(f"HATA: {e}")
        return {"fortune_text": f"Error: {str(e)}", "status": "error"}
