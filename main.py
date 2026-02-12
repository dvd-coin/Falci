from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

app = FastAPI()

# GÜVENLİK İZNİ (CORS) - Bunu eklemezsen Base44 bağlanamaz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tüm sitelere izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-fortune")
async def predict_fortune(
    images: List[UploadFile] = File(...), 
    language: str = Form("tr")
):
    # Senin fal bakma mantığın burada kalacak
    return {"fortune_text": "Falın bakılıyor...", "status": "success"}
