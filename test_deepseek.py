import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("CUDA available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"

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

prompt = "What are symptoms of diabetes?"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))