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
    language: str = Form(...) # Varsayılanı kaldırdık, Base44 göndermek ZORUNDA
):
    try:
        # 1. Resimleri Hazırla
        image_messages = []
        for image in images:
            content = await image.read()
            base64_image = base64.b64encode(content).decode('utf-8')
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # 2. Falcı Personası ve ÖRNEK FAL (Few-Shot Prompting)
        system_prompt = (
            f"Sen 'Firuze Abla' adında, hisleri çok kuvvetli, eski toprak bir falcısın. "
            f"KULLANICI DİLİ: {language}. Yanıtını SADECE bu dilde ver. Eğer 'English' ise İngilizce, 'Turkish' ise Türkçe konuş.\n\n"
            
            "TARZIN:\n"
            "- Asla 'fincanda şu var' deme. 'Ay içim daraldı', 'Yüreğin kabarmış', 'Bak bak şuraya bak' gibi girişler yap.\n"
            "- Kısa kesme. En az 3 paragraf dolusu hikaye anlat.\n"
            "- Aşk, para, kariyer ve sağlık konularına mutlaka değin.\n"
            "- Gizemli ol ama umut ver.\n\n"
            
            "ÖRNEK FAL ANLATIMI (BUNUN GİBİ KONUŞ):\n"
            "'Ay kuzum, senin yüreğin nasıl şişmiş böyle! Fincanı elime aldığım an bir ağırlık çöktü içime, sanki söylenmemiş sözler var boğazında düğümlenen. "
            "Bak şurada, fincanın dibinde kocaman bir balık var, görüyor musun? Bu balık nasip demek, kısmet demek! Hanene öyle temiz bir para girecek ki, o sıkıntılarını bir anda silip atacak. "
            "Ama tabağında sinsi bir göz var, sana hasetle bakan, yüzüne gülüp arkandan konuşan esmer bir kadın... Aman diyeyim, sırlarını herkese açma bu ara. "
            "Yolun var, çok aydınlık bir yol. Üç vakte kadar bir haber alacaksın ve sevinçten eteklerin zil çalacak. Hadi bakalım, niyetin kabul olsun!'"
        )

        # 3. Falı Yorumlat (GPT-4o)
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [*image_messages, {"type": "text", "text": "Hadi abla, yorumla bakalım ne görüyorsun?"}]}
            ],
            temperature=0.9,
            max_tokens=1000
        )
        
        fortune_text = completion.choices[0].message.content
        
        # 4. Sesi Üret (OpenAI TTS - Onyx veya Shimmer sesi)
        # Dil İngilizce ise ses tonu biraz daha farklı olabilir ama Shimmer genelde kadın falcıya uyar.
        audio_response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer", # 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
            input=fortune_text
        )
        
        # Sesi Base64'e çevirip gönderiyoruz
        audio_base64 = base64.b64encode(audio_response.content).decode('utf-8')
        
        return {
            "fortune_text": fortune_text, 
            "audio_base64": audio_base64,
            "status": "success"
        }

    except Exception as e:
        print(f"Hata: {e}")
        return {"fortune_text": f"Enerji hattında kopukluk var kuzum: {str(e)}", "status": "error"}
