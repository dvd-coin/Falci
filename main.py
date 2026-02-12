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
    print(f"--- GELEN DİL: {language} ---")

    # Dil Ayarı
    if language and "en" in language.lower():
        lang_instruction = "English"
        # İngilizce için dramatik giriş
        intro = "Oh my dear... Let me look closer at this cup..."
    else:
        lang_instruction = "Turkish"
        # Türkçe için dramatik giriş
        intro = "Ah canım benim... Gel bakayım şöyle yakına..."

    try:
        image_messages = []
        for image in images:
            content = await image.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # --- DUYGU VE TONLAMA İÇİN ÖZEL PROMPT ---
        system_prompt = (
            f"Sen mistik bir falcısın ama aynı zamanda bir tiyatrocusun. "
            f"DİL: {lang_instruction}. "
            f"KURALLAR:\n"
            f"1. Asla dümdüz okuma. Metnin içine '...' (üç nokta) koyarak bol bol ES VER. Bu, seslendirmede nefes almanı sağlar.\n"
            f"2. Duygu belirt! 'Hmm...', 'Ah!', 'Vay canına!' gibi ünlemler kullan.\n"
            f"3. Cümlelerin kısa olsun. Uzun cümleleri böl.\n"
            f"4. Robot gibi değil, sanki karşında en yakın arkadaşın varmış gibi fısıldayarak konuş.\n"
            f"5. Falı 3 kısa paragraf yap ama çok etkileyici olsun."
        )

        # Chat GPT'ye isteği gönder
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [*image_messages, {"type": "text", "text": "Hadi yorumla."}]}
            ],
            max_tokens=600 
        )
        
        # Giriş cümlesini ekleyerek metni oluştur (Seslendirme için)
        raw_text = response.choices[0].message.content
        full_text = f"{intro} {raw_text}"
        
        # SES ÜRETİMİ (TTS-1)
        # 'Nova' sesi biraz daha enerjik ve doğaldır. 'Shimmer' daha buğuludur.
        audio_response = client.audio.speech.create(
            model="tts-1",
            voice="nova", 
            input=full_text,
            response_format="mp3"
        )
        
        audio_base64 = base64.b64encode(audio_response.content).decode('utf-8')
        
        return {
            "fortune_text": full_text, 
            "audio_base64": audio_base64,
            "status": "success"
        }

    except Exception as e:
        print(f"HATA: {e}")
        return {"fortune_text": "Error", "status": "error"}
