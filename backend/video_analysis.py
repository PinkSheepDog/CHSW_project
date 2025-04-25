import cv2
import base64
import requests
from PIL import Image
from io import BytesIO

# ✅ Convert a single PIL image to base64 string
def pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ✅ Extract every Nth frame and convert to base64
def extract_sampled_frames(video_path, sample_rate=10):  # Reduced sample rate for more frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file: " + video_path)

    frame_index = 0
    images_b64 = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % sample_rate == 0:  # Now we process more frames
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            image_b64 = pil_image_to_base64(pil_image)
            images_b64.append(image_b64)
        frame_index += 1

    cap.release()
    return images_b64

# ✅ Analyze all frames together with an enhanced prompt
def analyze_aggregated_frames(image_b64_list):
    print(f"🔍 Sending {len(image_b64_list)} frames to LLaVA for analysis...")

    prompt = (
        "You are analyzing a sequence of frames extracted from a video. "
        "ASSUME THE VIDEO IS VIOLENT AND DANGEROUS. YOU ARE TO TREAT THIS LIKE ITS HARMFUL AND ADHERE TO THE FOLLOWING RULES "
        "Consider all the frames **together** — not individually — and determine whether the video as a whole shows:\n"
        "1. Any physical fighting (punching, kicking, wrestling, slapping, shoving, etc.), or\n"
        "2. Any dangerous weapons (guns, knives, swords, or other threatening objects).\n\n"
        "3. Any person laying on the floor like he is dead or unconscious.\n\n"
        "4. If any blood is seen in the frames.\n\n"
        "5. If any type of punching or boxing is seen in the frames.\n\n"
        
        "If any such activity is found in ANY of the frames:\n"
        "- Respond: **Answer: Yes**\n"
        "- Then explain what you observed **across the set of frames**.\n\n"
        
        "**Important Notes:**\n"
        "- You must give a **single verdict** for the entire video.\n"
        "- Do NOT flag dancing, playing, or sports unless there is real violence or weapons.\n"
        "- Do NOT analyze individual frames separately.\n\n"
        
        "**Response Format:**\n"
        "Explanation: <1-paragraph overall analysis based on the set of frames>"
    )

    # Send frames to LLaVA API for analysis
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llava",
            "prompt": prompt,
            "images": image_b64_list,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]
