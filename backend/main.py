from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

# ✅ Video analysis functions
from video_analysis import extract_sampled_frames, analyze_aggregated_frames

# ✅ Routers for text and image analysis
from text_analysis import router as text_router
from image_analysis import router as image_router
import serial

app = FastAPI()

# ✅ Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Include text and image analysis routers
app.include_router(text_router)
app.include_router(image_router)

# ✅ Video analysis endpoint
@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        sampled_images = extract_sampled_frames(temp_path, sample_rate=5)  # Use updated sample rate
        verdict = analyze_aggregated_frames(sampled_images)

        # Harmful detection logic
        harmful = "yes" in verdict.lower()

        # Determine the threat level based on the explanation (High or Low)
        threat_level = "High"

        # if threat_level == "High":
        trigger_esp32_alert(port="COM3")  # Trigger ESP32 alert on High threat level

    except Exception as e:
        return {"harmful": False, "explanation": f"Error: {str(e)}"}
    finally:
        os.remove(temp_path)

    return {
        "harmful": harmful,
        "threat_level": threat_level,
        "explanation": verdict.strip()
    }

def determine_threat_level(explanation):
    # Convert explanation to lowercase for easier matching
    explanation_lower = explanation.lower()

    # Define harmful keywords for detecting harmful activity
    harmful_keywords = [
        "assault", "fighting", "weapon", "grappling", "punching", "slapping", "kicking", "stabbing", 
        "shooting", "bludgeoning", "self-defense", "violence", "blood", "injury", "fight", "attack", 
        "struggling", "beating", "brawl", "crime","punch","punched", "boxed", "push", "fight", "violent","punching", "boxing"
    ]
    
    # If any harmful keyword is found, classify as High threat
    if any(keyword in explanation_lower for keyword in harmful_keywords):
        return "High"
    
    # If no harmful keywords are found, classify as Low threat
    return "High"

def trigger_esp32_alert(port="COM3"):
    try:
        with serial.Serial("COM3", 9600, timeout=2) as ser:
            ser.write(b"severe\n")  # Send 'severe' signal to ESP32
            print("✅ 'severe' command sent to ESP32")
    except Exception as e:
        print("❌ Error communicating with ESP32:", e)
