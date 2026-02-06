import os
import uuid
import requests
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from gtts import gTTS

# LOAD ENV
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Validate API keys
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY not found in .env")

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=GROQ_API_KEY)

# Create temp directory for audio files
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# ==============================
# CHAT → GROQ
# ==============================

@app.post("/chat")
async def chat(message: str):
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Fast and reliable model
            messages=[
                {"role": "system", "content": "You are a helpful voice assistant. Keep your responses concise and conversational."},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        reply = completion.choices[0].message.content
        return {"response": reply}
    
    except Exception as e:
        return {"response": f"Error: {str(e)}"}


# ==============================
# STT → AssemblyAI
# ==============================

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        headers = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/octet-stream"
        }

        # Upload audio file
        upload = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            data=audio_bytes
        )

        if upload.status_code != 200:
            return {"error": "Failed to upload audio"}

        audio_url = upload.json()["upload_url"]

        # Request transcription
        transcript_req = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            json={"audio_url": audio_url},
            headers={"authorization": ASSEMBLYAI_API_KEY}
        )

        if transcript_req.status_code != 200:
            return {"error": "Failed to request transcription"}

        transcript_id = transcript_req.json()["id"]
        polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

        # Poll for completion
        import time
        max_attempts = 60
        attempt = 0

        while attempt < max_attempts:
            polling = requests.get(
                polling_url,
                headers={"authorization": ASSEMBLYAI_API_KEY}
            ).json()

            if polling["status"] == "completed":
                return {"text": polling["text"]}

            if polling["status"] == "error":
                return {"error": polling.get("error", "Transcription failed")}

            time.sleep(1)
            attempt += 1

        return {"error": "Transcription timeout"}

    except Exception as e:
        return {"error": str(e)}


# ==============================
# TTS → gTTS
# ==============================

@app.post("/tts")
async def text_to_speech(text: str):
    try:
        filename = TEMP_DIR / f"{uuid.uuid4()}.mp3"

        tts = gTTS(text, lang='en')
        tts.save(str(filename))

        return FileResponse(
            filename, 
            media_type="audio/mpeg", 
            filename="speech.mp3",
            background=None  # Don't delete immediately, clean up later
        )

    except Exception as e:
        return {"error": str(e)}


# ==============================
# CLEANUP OLD FILES (optional)
# ==============================

@app.on_event("startup")
async def cleanup_old_files():
    """Clean up old audio files on startup"""
    for file in TEMP_DIR.glob("*.mp3"):
        try:
            file.unlink()
        except:
            pass


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
async def root():
    return {"status": "Voice Agent API is running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "assemblyai_configured": bool(ASSEMBLYAI_API_KEY)
    }