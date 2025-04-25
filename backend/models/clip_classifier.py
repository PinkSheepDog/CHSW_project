
import torch
from PIL import Image
from torchvision import transforms

# Path to your saved TorchScript model
CLIP_MODEL_PATH = r"C:\CODING\Hackathon\harmful_meme_detector_optimized_CLIP.pt"

DEVICE = torch.device("cpu")  # <-- Force CPU


# Image preprocessing (must match training config exactly)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

print("[DEBUG] Loading CLIP model...")
# Force load to CPU
model = torch.jit.load(CLIP_MODEL_PATH, map_location="cpu").to("cpu")

model.eval()
print("[DEBUG] CLIP model loaded successfully.")


def classify_clip(text: str, image_path: str) -> tuple[bool, float]:
    """
    Runs CLIP classifier on the given image (text is ignored for now).
    Returns (is_harmful: bool, raw_score: float)
    """
    print(f"[DEBUG] classify_clip() called with:")
    print(f"        Image path: {image_path}")
    print(f"        Caption: {text}")

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            score = model(image_tensor).item()

        is_harmful = score > 0.5

        print(f"[DEBUG] Raw model score = {score:.4f}")
        print(f"[DEBUG] Threshold = 0.5")
        print(f"[DEBUG] Classification result: {'HARMFUL' if is_harmful else 'SAFE'}")

        return is_harmful, score
    except Exception as e:
        print(f"[ERROR] Failed during CLIP inference: {e}")
        return False, 0.0