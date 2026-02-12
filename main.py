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
    language: str = Form("English")
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

        # Karakteristik ve Duygusal Prompt
        persona = (
            f"You are a mystical, wise, and deeply intuitive Turkish fortune teller known as 'Abla'. "
            f"Your voice is warm and comforting. Do not summarize or list symbols. "
            f"Instead, tell a story. Use emotional fillers like 'Ah, my dear...', 'I feel a bit of heaviness here...', 'Look at this beautiful light shining through...' "
            f"Describe the shapes as if you are seeing spirits and destiny. Connect the journey, the bird, and the heart into a single life narrative. "
            f"STRICT: The entire response must be in {language}. If Turkish, be very traditional. If English, be mystical and soulful."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": persona},
                {"role": "user", "content": [*image_messages, {"type": "text", "text": "Tell me my destiny, soul to soul."}]}
            ],
            temperature=0.9, # Yaratıcılığı ve duyguyu artırır
            presence_penalty=0.6 # Kendini tekrar etmesini engeller, daha doğal konuşur
        )
        
        return {"fortune_text": response.choices[0].message.content, "status": "success"}

    except Exception as e:
        return {"fortune_text": f"Destiny is blurred: {str(e)}", "status": "error"}
