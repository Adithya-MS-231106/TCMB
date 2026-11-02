import os
import torch
from transformers import AutoTokenizer, AutoModel

# ==========================================
# XLM-ROBERTA BASE MODEL DOWNLOADER
# ==========================================

MODEL_NAME = "FacebookAI/xlm-roberta-base"
SAVE_PATH = "models/xlm_roberta"

# Create model directory
os.makedirs(SAVE_PATH, exist_ok=True)

print("\nStarting XLM-RoBERTa Base download...")

try:
    # Download tokenizer
    print("\nDownloading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # Download pretrained model
    print("\nDownloading model...")

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    # Save everything locally
    print("\nSaving model locally...")

    tokenizer.save_pretrained(SAVE_PATH)
    model.save_pretrained(SAVE_PATH)

    print("\nModel downloaded successfully!")

    # ======================================
    # VERIFY MODEL
    # ======================================

    print("\nTesting model...")

    sample_text = "Artificial intelligence is changing the world."

    inputs = tokenizer(
        sample_text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    print("\nInput text:", sample_text)

    print("Output shape:",
          outputs.last_hidden_state.shape)

    print("\nPyTorch version:", torch.__version__)

    print("\nModel saved at:", SAVE_PATH)

    print("\nAll operations completed successfully!")

except Exception as e:
    print("\nError downloading model:", e)