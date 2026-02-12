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
    language: str = Form("Turkish") # Varsayılan dili belirledik
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

        # Samimi ve Doğal Falcı Karakteri
        persona_prompt = (
            f"Sen çok deneyimli, gizemli ve bir o kadar da samimi bir Türk kahvesi falcısısın. "
            f"Resimlere bakarken 'İlk fincan, ikinci tabağın üstü' gibi teknik terimler kullanma. "
            f"Onun yerine 'Aman yarabbim, şuraya bak...', 'Gönlün biraz daralmış sanki...', 'Bak şurada bir kısmetin var' gibi gerçekçi bir dil kullan. "
            f"Falın gidişatı akıcı olsun, formaliteden uzak dur. Şekilleri hayatın içinden hikayelerle bağla. "
            f"ÖNEMLİ: Falı mutlaka ve sadece {language} dilinde anlat. Eğer dil English seçildiyse mutlaka İngilizce konuş."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": persona_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Canım falcı, bak bakalım fincanımda neler var?"},
                        *image_messages 
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.85 # Daha doğal ve öngörülemez konuşması için artırdık
        )
        
        return {"fortune_text": response.choices[0].message.content, "status": "success"}

    except Exception as e:
        return {"fortune_text": f"Hata oluştu canım: {str(e)}", "status": "error"}
