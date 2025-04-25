from fastapi import APIRouter, Form
import requests

router = APIRouter()

@router.post("/analyze-text")
async def analyze_text(text: str = Form(...)):
    prompt = (
        "You are a content moderation expert. Carefully analyze the following user-generated text:\n\n"
        f"{text}\n\n"
        "1. Determine if the text contains hate speech, threats, offensive language, or discrimination.\n"
        "2. Respond with 'Yes' if the content is harmful, or 'No' if it is not.\n"
        "3. Then, in a separate paragraph, explain your reasoning in detail using natural language.\n"
        "Format: Answer (Yes/No) + Paragraph explanation."
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "openhermes", "prompt": prompt, "stream": False}
    )

    result = response.json().get("response", "").strip()

    return {
        "harmful": "yes" in result.lower(),
        "explanation": result  # Let frontend display the full paragraph
    }
