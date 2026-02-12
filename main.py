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
    language: str = Form(...) # Base44'ten gelmesi zorunlu
):
    # 1. LOGLAMA: Base44'ün ne gönderdiğini Railway Loglarında göreceğiz
    print(f"GELEN DİL İSTEĞİ: {language}")

    # 2. DİLİ STANDARTLAŞTIRMA (Ne gelirse gelsin doğruya çevir)
    lang_lower = language.lower()
    if any(x in lang_lower for x in ['en', 'eng', 'ing', 'usa']):
        selected_lang = "English"
        voice_model = "shimmer" # İngilizceye daha uygun bir ton
    else:
        selected_lang = "Turkish"
        voice_model = "onyx" # Türkçeye biraz daha tok giden ses (veya shimmer kalabilir)

    try:
        # Resimleri Hazırla
        image_messages = []
        for image in images:
            content = await image.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # Karakter ve Dil Promptu (Kesinleştirilmiş)
        system_prompt = (
            f"Sen 'Firuze Abla' adında, hisleri kuvvetli bir falcısın. "
            f"ŞU AN GEÇERLİ DİL: {selected_lang.upper()}. "
            f"Yanıtını SADECE {selected_lang} dilinde ver. Başka dil kullanma.\n\n"
            f"Eğer dil English ise: 'Oh my dear, let me look at your cup...' diye başla.\n"
            f"Eğer dil Turkish ise: 'Ay canım, içim şişti fincana bakınca...' diye başla.\n\n"
            "Falı mistik, samimi ve hikaye anlatır gibi yorumla. En az 200 kelime olsun."
        )

        # AI'ya Gönder
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [*image_messages, {"type": "text", "text": "Yorumla abla."}]}
            ],
            temperature=0.8,
            max_tokens=800
        )
        
        fortune_text = completion.choices[0].message.content
        
        # Sesi Üret
        audio_response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=fortune_text
        )
        
        audio_base64 = base64.b64encode(audio_response.content).decode('utf-8')
        
        return {
            "fortune_text": fortune_text, 
            "audio_base64": audio_base64,
            "status": "success",
            "detected_lang": selected_lang # Base44'te debug için geri gönderiyoruz
        }

    except Exception as e:
        print(f"Hata: {e}")
        return {"fortune_text": f"Error: {str(e)}", "status": "error"}
