from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "models/xlm_roberta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)

text = "Welcome to our multilingual NLP project."

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

print("Model loaded successfully!")
print("Token IDs:", inputs["input_ids"])
print("Output shape:", outputs.last_hidden_state.shape)