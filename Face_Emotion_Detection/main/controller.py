from fastapi import FastAPI
from realtimeemotiondetection import detect_emotion
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Only allow this frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get('/detect_emotion')
def get_emotion_from_face():
    captured_emotion = detect_emotion()
    print(captured_emotion)
    return {'Dominant emotion':captured_emotion}

