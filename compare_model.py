from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "bigscience/bloom-560m"
dpo_model_path = "./dpo-medical-model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading ORIGINAL model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading DPO model...")
dpo_model = AutoModelForCausalLM.from_pretrained(
    dpo_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate(model, question):
    prompt = "Jawab pertanyaan medis berikut dalam bahasa Indonesia: " + question

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace(prompt, "").strip()

# TEST QUESTIONS
questions = [
    "Apa penyebab flu?",
    "Bagaimana cara mengobati demam?",
    "Apa gejala diabetes?",
    "Kapan harus ke dokter saat sakit kepala?"
]

for q in questions:
    print("\n==============================")
    print("QUESTION:", q)

    print("\n--- BASE MODEL ---")
    print(generate(base_model, q))

    print("\n--- DPO MODEL ---")
    print(generate(dpo_model, q))