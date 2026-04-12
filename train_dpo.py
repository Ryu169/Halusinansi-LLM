import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

# =========================
# CONFIG
# =========================
model_name = "deepseek-ai/deepseek-llm-7b-base"
dataset_path = "dpo_dataset.jsonl"

print("🚀 Starting DPO Training for DeepSeek")

# =========================
# TOKENIZER
# =========================
print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# =========================
# 4-BIT CONFIG (WAJIB)
# =========================
print("⚙️ Setting 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# =========================
# LOAD MODEL
# =========================
print("🧠 Loading DeepSeek model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# IMPORTANT
model.config.use_cache = False

# =========================
# LOAD DATASET
# =========================
print("📂 Loading dataset...")
dataset = load_dataset("json", data_files=dataset_path)

def format_data(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dataset = dataset.map(format_data)

# =========================
# LORA CONFIG (DEEPSEEK)
# =========================
print("🔧 Setting LoRA config...")

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],  # 🔥 cocok untuk DeepSeek
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# =========================
# TRAINING CONFIG (OPTIMAL 8GB GPU)
# =========================
print("📊 Setting training config...")

training_args = DPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # hemat VRAM
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=5,
    save_steps=50,
    output_dir="./dpo_output",
    fp16=True,
    report_to="none"
)

# =========================
# TRAINER
# =========================
print("🤖 Initializing trainer...")

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    peft_config=peft_config
)

# =========================
# TRAINING
# =========================
print("🔥 Training started...")
trainer.train()

# =========================
# SAVE MODEL
# =========================
print("💾 Saving model...")
trainer.save_model("dpo-deepseek-model")

print("✅ TRAINING COMPLETE")