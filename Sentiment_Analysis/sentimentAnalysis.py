import pickle
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for frontend (http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domain in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

print(os.getcwd())
with open('random_forest_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)




# Download Vader Lexicon (only required once)
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

def get_emotion_score(text):
    vader_score = analyzer.polarity_scores(text)['compound']  # VADER sentiment
    blob_score = TextBlob(text).sentiment.polarity  # TextBlob sentiment
    return (vader_score + blob_score) / 2  # Average score



def get_routine_score(text):    
    positive_words = ["routine", "schedule", "consistent", "habit", "exercise", "sleep", "work"]
    negative_words = ["irregular", "unstable", "disorganized", "lack of routine", "unstructured"]

    pos_count = sum(word in text.lower() for word in positive_words)
    neg_count = sum(word in text.lower() for word in negative_words)

    score = (pos_count - neg_count) / (pos_count + neg_count + 1)  # Normalized score
    return max(0, min(1, score))  # Ensure score is between 0 and 1


def get_trauma_score(text):
    trauma_words = {
        "abuse": 9000, "violence": 8500, "accident": 7500, "death": 8000,
        "loss": 7000, "trauma": 9500, "assault": 9700, "grief": 7200
    }

    score = sum(trauma_words[word] for word in text.lower().split() if word in trauma_words)
    return min(10000, score)  # Cap score at 10,000


def getPrediction(res):
    trauma_score = get_trauma_score(text=res)
    emotion_score = get_emotion_score(text=res)
    routine_score = get_routine_score(text=res)
    features = pd.DataFrame([[emotion_score, routine_score, trauma_score]], 
                        columns=["Emotion_Score", "Routine_Score", "Trauma_Score"])
    # scaler.feature_names_in_ = None  # Ignore feature names
    scaled_features = scaler.transform(features)
    prediction = classifier.predict(scaled_features)
    return int(prediction[0])

class InputData(BaseModel):
    text: str  

@app.options("/predict")
async def options_predict():
    return JSONResponse(content=None, headers={"Access-Control-Allow-Origin": "*", 
                                               "Access-Control-Allow-Methods": "POST, OPTIONS",
                                               "Access-Control-Allow-Headers": "*"})
@app.post("/predict")
def predict(data: InputData):
    prediction = getPrediction(data.text)
    return prediction

print(f'Phase:{getPrediction('I am in trauma')}')


