import traceback
from huggingface_hub import InferenceClient

# Token Anda
HF_TOKEN = "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"
client = InferenceClient(api_key=HF_TOKEN)

model_id = "meta-llama/Llama-3.2-1B-Instruct"

try:
    print(f"Mencoba menghubungi model {model_id}...")
    
    # Format input diubah menjadi list of messages (gaya chat)
    messages = [
        {"role": "user", "content": "Apa gejala diabetes?"}
    ]
    
    # Menggunakan chat_completion, bukan text_generation
    response = client.chat_completion(
        messages=messages,
        model=model_id,
        max_tokens=500
    )
    
    print("\n--- Hasil Jawaban ---")
    # Cara mengekstrak teks jawaban dari response JSON-nya
    print(response.choices[0].message.content)

except Exception as e:
    print("\n=== ERROR DETAIL ===")
    print(f"Tipe Error: {type(e).__name__}")
    print(f"Pesan: {str(e)}")
    print("Traceback Lengkap:")
    traceback.print_exc()