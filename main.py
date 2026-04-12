from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from fastapi.responses import FileResponse

import datetime
import requests
import pandas as pd
import re
import json
import os

# ==============================
# OPENROUTER API CONFIG (DEEPSEEK)
# ==============================
# Ganti teks di bawah ini dengan API Key OpenRouter Anda!
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Menggunakan model DeepSeek R1 versi 8B
MODEL_ID = "deepseek/deepseek-chat"

def generate_answer(prompt: str) -> str:
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        result = response.json()
        
        # Mengecek apakah ada error dari OpenRouter (misal: saldo habis / token salah)
        if "error" in result:
             return f"[API Error] {result['error']['message']}"
             
        # Mengembalikan teks jawaban
        return result['choices'][0]['message']['content']

    except Exception as e:
        return f"[System Error] {str(e)}"


# ==============================
# DATABASE SETUP
# ==============================

DATABASE_URL = "sqlite:///qa_dataset.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class QAData(Base):
    __tablename__ = "qa_data"

    id = Column(Integer, primary_key=True)
    question = Column(String)
    answer = Column(String)
    entropy = Column(Float)
    risk = Column(String)
    timestamp = Column(String)

class PreferenceData(Base):
    __tablename__ = "preference_data"

    id = Column(Integer, primary_key=True)
    question = Column(String)
    answer = Column(String)
    entropy = Column(Float)
    label = Column(String)
    timestamp = Column(String)

Base.metadata.create_all(engine)

# ==============================
# FASTAPI SETUP
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

class Evaluation(BaseModel):
    question: str
    answer: str
    entropy: float
    label: str

class PreferencePair(BaseModel):
    question: str
    chosen: str
    rejected: str

# ==============================
# LOAD DATASET EXCEL
# ==============================

DATASET_PATH = "preprocessing_semua_dokter_clean.xlsx"

df = pd.read_excel(DATASET_PATH)

df.columns = df.columns.str.lower()
df["question"] = df["question"].astype(str).str.strip().str.lower()
df["answer"]   = df["answer"].astype(str).str.strip()
df["doktor"]   = df["doktor"].astype(str).str.strip()

qa_dict = dict(zip(df["question"], df["answer"]))

print("✅ Dataset loaded:", len(df))


# ==============================
# QA ENDPOINT
# ==============================

@app.post("/ask")
def ask(q: Question):

    db = SessionLocal()
    user_q_clean = q.question.strip().lower()

    # =========================
    # CHECK PREFERENCE OVERRIDE
    # =========================
    preferences = db.query(PreferenceData).filter(
        PreferenceData.question == user_q_clean
    ).all()

    chosen_answer = None
    for p in preferences:
        if p.label == "chosen":
            chosen_answer = p.answer

    if chosen_answer:
        db.close()
        return {
            "question": q.question,
            "model_answer": chosen_answer,
            "entropy": 0.0,
            "risk": "Overridden (Doctor Preference)",
            "source": "doctor"
        }

    db.close()

    # =========================
    # EXACT MATCH DATASET
    # =========================
    if user_q_clean not in qa_dict:
        return {"error": "Pertanyaan tidak ditemukan di dataset"}

    filtered = df[df["question"] == user_q_clean]

    if filtered.empty:
        return {"error": "Data tidak ditemukan"}

    row          = filtered.iloc[0]
    ground_truth = row["answer"]
    doktor       = row["doktor"]

    # =========================
    # PROMPT
    # =========================
    prompt = (
        "Anda adalah dokter profesional.\n\n"
        "Jawaban HARUS:\n"
        "- berdasarkan pengetahuan medis\n"
        "- maksimal 2 kalimat\n"
        "- tidak boleh membuat daftar\n"
        "- langsung ke inti\n\n"
        f"Pertanyaan: {q.question}\n"
        "Jawaban:"
    )

    # =========================
    # GENERATE VIA HF API
    # =========================
    raw_text = generate_answer(prompt)

    # =========================
    # CLEAN OUTPUT
    # =========================
    if "Jawaban:" in raw_text:
        raw_text = raw_text.split("Jawaban:")[-1]

    raw_text  = raw_text.strip()
    raw_text  = re.sub(r"^\s*\d+[\.\)]\s*", "", raw_text)   # hapus numbering
    raw_text  = re.sub(r"\s+", " ", raw_text).strip()
    raw_text  = raw_text.replace("..", ".")

    sentences = re.split(r'(?<=[.!?])\s+', raw_text)
    text      = " ".join(sentences[:2])                      # max 2 kalimat

    # =========================
    # ENTROPY — tidak tersedia via API
    # =========================
    entropy = 0.0
    risk    = "N/A (API Mode)"

    # =========================
    # SAVE TO DATABASE
    # =========================
    db = SessionLocal()

    data = QAData(
        question  = q.question,
        answer    = text,
        entropy   = entropy,
        risk      = risk,
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    db.add(data)
    db.commit()
    db.close()

    # =========================
    # RETURN
    # =========================
    return {
        "question":     q.question,
        "model_answer": text,
        "ground_truth": ground_truth,
        "doktor":       doktor,
        "entropy":      entropy,
        "risk":         risk
    }


# ==============================
# DATASET ENDPOINT
# ==============================

@app.get("/dataset")
def get_dataset():

    db   = SessionLocal()
    data = db.query(QAData).all()
    db.close()

    result = []
    for d in data:
        question_clean = d.question.strip().lower()
        dataset_answer = qa_dict.get(question_clean, "Tidak ditemukan di dataset")
        result.append({
            "question":       d.question,
            "model_answer":   d.answer,
            "dataset_answer": dataset_answer,
            "entropy":        d.entropy,
            "risk":           d.risk,
            "timestamp":      d.timestamp
        })

    return result


@app.get("/hitl_queue")
def get_hitl_queue():

    db   = SessionLocal()
    data = db.query(QAData).filter(QAData.entropy >= 2.5).all()
    db.close()

    return [{
        "id":        d.id,
        "question":  d.question,
        "answer":    d.answer,
        "entropy":   d.entropy,
        "risk":      d.risk,
        "timestamp": d.timestamp
    } for d in data]


# ==============================
# PREFERENCE ENDPOINTS
# ==============================

@app.post("/save_preference_pair")
def save_preference_pair(data: PreferencePair):

    db = SessionLocal()

    db.query(PreferenceData).filter(
        PreferenceData.question == data.question.strip().lower(),
        PreferenceData.label.in_(["chosen", "rejected"])
    ).delete()

    db.add(PreferenceData(
        question = data.question.strip().lower(),
        answer   = data.chosen,
        entropy  = 0,
        label    = "chosen"
    ))

    db.add(PreferenceData(
        question = data.question.strip().lower(),
        answer   = data.rejected,
        entropy  = 0,
        label    = "rejected"
    ))

    db.commit()
    db.close()

    return {"status": "saved"}


@app.post("/evaluate")
def evaluate(data: Evaluation):

    db = SessionLocal()

    db.query(PreferenceData).filter(
        PreferenceData.question == data.question,
        PreferenceData.label.in_(["E", "N", "C"])
    ).delete()

    db.add(PreferenceData(
        question  = data.question.strip().lower(),
        answer    = data.answer,
        entropy   = data.entropy,
        label     = data.label,
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    db.commit()
    db.close()

    return {"status": "saved"}


@app.get("/preferences")
def preferences():

    db   = SessionLocal()
    data = db.query(PreferenceData).all()
    db.close()

    return [{
        "question": d.question,
        "answer":   d.answer,
        "entropy":  d.entropy,
        "label":    d.label
    } for d in data]


# ==============================
# EXPORT ENDPOINTS
# ==============================

def build_dpo_dataset(db):
    data    = db.query(PreferenceData).all()
    prompts = {}

    for d in data:
        if d.question not in prompts:
            prompts[d.question] = {"chosen": None, "rejected": None}
        if d.label == "chosen":
            prompts[d.question]["chosen"]   = d.answer
        if d.label == "rejected":
            prompts[d.question]["rejected"] = d.answer

    return [
        {"prompt": q, "chosen": v["chosen"], "rejected": v["rejected"]}
        for q, v in prompts.items()
        if v["chosen"] and v["rejected"]
    ]


@app.get("/export_dpo")
def export_dpo():
    db      = SessionLocal()
    dataset = build_dpo_dataset(db)
    db.close()
    return dataset


@app.get("/export_dpo_file")
def export_dpo_file():
    db      = SessionLocal()
    dataset = build_dpo_dataset(db)
    db.close()

    with open("dpo_dataset.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    return {"status": "dataset exported", "total": len(dataset)}


# ==============================
# DOWNLOAD EXCEL
# ==============================

@app.get("/download_qa_excel")
def download_qa_excel():

    db   = SessionLocal()
    data = db.query(QAData).all()
    db.close()

    df_out = pd.DataFrame([{
        "Question":  d.question,
        "Answer":    d.answer,
        "Entropy":   d.entropy,
        "Risk":      d.risk,
        "Timestamp": d.timestamp
    } for d in data])

    path = "qa_dataset.xlsx"
    df_out.to_excel(path, index=False)
    return FileResponse(path, filename="qa_dataset.xlsx")


@app.get("/download_preferences_excel")
def download_preferences_excel():

    db   = SessionLocal()
    data = db.query(PreferenceData).all()
    db.close()

    df_out = pd.DataFrame([{
        "Question": d.question,
        "Answer":   d.answer,
        "Entropy":  d.entropy,
        "Label":    d.label
    } for d in data])

    path = "preference_dataset.xlsx"
    df_out.to_excel(path, index=False)
    return FileResponse(path, filename="preference_dataset.xlsx")


# ==============================
# EVALUATION STATS
# ==============================

@app.get("/evaluation_stats")
def evaluation_stats():

    db   = SessionLocal()
    data = db.query(PreferenceData).all()
    db.close()

    entropies  = [d.entropy for d in data if d.entropy is not None]
    avg_entropy = round(sum(entropies) / len(entropies), 3) if entropies else 0

    return {
        "total":       len(data),
        "E":           sum(1 for d in data if d.label == "E"),
        "N":           sum(1 for d in data if d.label == "N"),
        "C":           sum(1 for d in data if d.label == "C"),
        "avg_entropy": avg_entropy
    }
