import pandas as pd
import datetime
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from main import SessionLocal, QAData


print("Loading dataset...")

df = pd.read_csv("Medical_QA_Cleaned.csv")

# =========================
# DATASET PREPROCESSING
# =========================

# hapus pertanyaan kosong
df = df.dropna(subset=["question"])

# ubah ke string
df["question"] = df["question"].astype(str)

# hapus spasi berlebih
df["question"] = df["question"].str.strip()

# hapus pertanyaan terlalu pendek
df = df[df["question"].str.len() > 5]

# hapus duplicate
df = df.drop_duplicates(subset=["question"])

df = df.reset_index(drop=True)

print("Total data setelah cleaning:", len(df))


print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")


# =========================
# ENTROPY FUNCTION
# =========================

def calculate_entropy(scores):

    entropies = []

    for logits in scores:

        probs = F.softmax(logits, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        entropies.append(entropy.mean().item())

    return sum(entropies) / len(entropies)


db = SessionLocal()

print("Start generating answers...")

for i, row in df.iterrows():

    try:

        question = str(row["question"])

        if question.strip() == "" or question == "nan":
            continue

        prompt = "Jawab pertanyaan medis berikut dalam bahasa Indonesia: " + question

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            return_dict_in_generate=True,
            output_scores=True
        )

        generated_tokens = outputs.sequences[0]

        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        answer = answer.replace(prompt, "").strip()

        entropy = calculate_entropy(outputs.scores)

        entropy = round(entropy, 3)

        if entropy < 1.5:
            risk = "Low Risk"
        elif entropy < 2.5:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        data = QAData(
            question=question,
            answer=answer,
            entropy=entropy,
            risk=risk,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        db.add(data)

        if i % 10 == 0:
            print("Processed:", i)

    except Exception as e:

        print("Error pada index", i)
        print(e)

db.commit()
db.close()

print("Dataset ingestion selesai")