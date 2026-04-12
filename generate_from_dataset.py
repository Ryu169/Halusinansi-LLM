import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import QAData, Base  # pakai model DB kamu
import datetime

# ======================
# LOAD MODEL
# ======================
model_name = "deepseek-ai/deepseek-llm-7b-base"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# DATABASE
# ======================
engine = create_engine("sqlite:///qa_dataset.db")
SessionLocal = sessionmaker(bind=engine)

# ======================
# ENTROPY
# ======================
def calculate_entropy(scores):
    entropies = []
    for logits in scores:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        entropies.append(entropy.mean().item())
    return sum(entropies) / len(entropies)

# ======================
# LOAD DATASET
# ======================
df = pd.read_csv("Medical_QA_Cleaned.csv")

# pastikan kolom: question
questions = df["question"].tolist()

# ======================
# GENERATE LOOP
# ======================
db = SessionLocal()

for q in questions[:50]:  # batasi dulu (testing)

    prompt = f"Pertanyaan: {q}\nJawaban:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        return_dict_in_generate=True,
        output_scores=True
    )

    entropy = round(calculate_entropy(outputs.scores), 3)

    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    if "Jawaban:" in text:
        text = text.split("Jawaban:")[-1].strip()

    text = " ".join(text.split())

    # Risk
    if entropy < 1.5:
        risk = "Low"
    elif entropy < 2.5:
        risk = "Medium"
    else:
        risk = "High"

    data = QAData(
        question=q,
        answer=text,
        entropy=entropy,
        risk=risk,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    db.add(data)
    print(f"✔ {q[:40]} | entropy={entropy}")

db.commit()
db.close()

print("✅ DONE GENERATE")