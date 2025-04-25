from fastapi import APIRouter, UploadFile, File
import base64
import requests

router = APIRouter()

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    # Step 1: Read the image file
    image_bytes = await file.read()
    # Step 2: Convert image to base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Step 3: Construct the prompt for the moderation task
    prompt = (
        "You are a moderation assistant that evaluates images for potential harmful content based on established community safety guidelines. Please follow these instructions:\n\n"
        "DO NOT ASSUME JOKES OR HUMOR. TREAT THIS SERIOUSLY.\n\n"
        "1. Analyze the attached image carefully.\n"
        "2. Determine if the image contains any harmful content. Harmful content includes, but is not limited to:\n"
        "   - Graphic violence or gore\n"
        "   - Explicit sexual content or nudity\n"
        "   - Hate symbols or hate speech\n"
        "   - Racism, discrimination, or any form of harmful bias\n"
        "   - Self-harm, extremist, or other dangerous content\n"
        "3. If the image is harmful:\n"
        "   - Flag the image as harmful.\n"
        "   - Provide a detailed moderation message explaining specifically which elements of the image contribute to its harmful nature.\n"
        "4. If the image is not harmful:\n"
        "   - Simply state that the image isn't harmful.\n"
        "Make sure your output is clear and isn't too long."
    )

    # Step 4: Send the image and prompt to the LLaVA API
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava", 
                "prompt": prompt, 
                "images": [image_b64], 
                "stream": False
            }
        )

        # Check for successful response
        response.raise_for_status()

        # Step 5: Get the analysis result from the LLaVA API
        result = response.json().get("response", "").strip()

        # Step 6: Check for harmful content keywords
        harmful_keywords = ["hate speech", "violence", "racism", "discrimination", "self-harm", "extremist"]
        is_harmful = any(keyword in result.lower() for keyword in harmful_keywords)

        # Set the threat level based on the content
        threat_level = "Low"
        if "hate speech" in result.lower() or "violence" in result.lower():
            threat_level = "High"
        elif "racism" in result.lower() or "discrimination" in result.lower():
            threat_level = "Medium"

        return {
            "harmful": is_harmful,
            "threat_level": threat_level,
            "explanation": result
        }

    except requests.exceptions.RequestException as e:
        # Handle errors like connection problems or invalid responses
        return {"harmful": False, "explanation": f"Error in analysis: {e}"}
