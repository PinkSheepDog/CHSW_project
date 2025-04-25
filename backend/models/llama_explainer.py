import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests

def generate_explanation(reason: str) -> str:
    print("[DEBUG] Calling OpenHermes via Ollama...")
    prompt = (
        f"This content was flagged as '{reason}'. "
        f"Please generate a short, user-friendly moderation message explaining why it might be harmful, "
        f"and how the user could improve their content to meet community standards."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "openhermes",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json().get("response", "[ERROR] No response from OpenHermes.")
    except Exception as e:
        print(f"[ERROR] OpenHermes call failed: {e}")
        return "[ERROR] Failed to get explanation from OpenHermes."




LLaVA_MODEL_PATH = "C:/CODING/Hackathon/LLaVA/models/llava-1.5-7b-hf"

# Load model and processor once
print("[DEBUG] Loading LLaVA model...")
llava_device = "cuda" if torch.cuda.is_available() else "cpu"
llava_model = LlavaForConditionalGeneration.from_pretrained(LLaVA_MODEL_PATH, torch_dtype=torch.float16).to(llava_device)
llava_processor = AutoProcessor.from_pretrained(LLaVA_MODEL_PATH)
print("[DEBUG] LLaVA model loaded successfully.")

def generate_llava_explanation(image_path: str, reason: str = "") -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = f"<image>\nUSER: This content was flagged as harmful. {reason} Please have a look at the content and figure out why exactly it was deemed harmful. if unsure, present a general moderation message indicating to the user that the content they posted is harmful .\nASSISTANT:"

        inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to(llava_device)
        #inputs = llava_processor(...)
        #inputs = {k: v.to(llava_device) for k, v in inputs.items()}

        output = llava_model.generate(**inputs, max_new_tokens=200)
        response = llava_processor.batch_decode(output, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        print(f"[ERROR] LLaVA explanation failed: {e}")
        return "LLaVA failed to generate explanation."



