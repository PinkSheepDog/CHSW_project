from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
import torch

# Path to your saved model
MODEL_PATH = r"C:\Python Scripts\Google_Script_Hackathon\Google-Hackathon-main\deberta_harmful_text_classifier (3)"

print("[DEBUG] Loading DeBERTa model from disk...")
model = DebertaV2ForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("[DEBUG] DeBERTa model loaded successfully.")


def classify_text(text: str) -> bool:
    print(f"[DEBUG] classify_text called with: {text}")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    print(f"[DEBUG] Model prediction: {predicted_class} (0 = not harmful, 1 = harmful)")

    return predicted_class == 1
