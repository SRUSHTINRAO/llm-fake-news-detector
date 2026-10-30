from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
import re

app = FastAPI()
client = Groq()

class NewsInput(BaseModel):
    text: str


# -----------------------------
# LLM Analysis Function
# -----------------------------
def analyze_with_llm(text):
    try:
        prompt = f"""
        You are a news credibility analyzer.

        Classify the following text strictly into one of:
        - Real
        - Fake
        - Uncertain

        Also explain briefly WHY.

        Return EXACT format:
        Prediction: Real OR Fake OR Uncertain
        Explanation: <2-3 lines>

        Text: {text}
        """

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content.strip()

        # ✅ STRICT parsing using regex
        match = re.search(r'Prediction:\s*(Real|Fake|Uncertain)', output, re.IGNORECASE)

        if match:
            prediction = match.group(1).capitalize()
        else:
            prediction = "Uncertain"

        return prediction, output

    except Exception as e:
        print("ERROR:", e)
        return "Uncertain", "Explanation not available"


# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: NewsInput):
    if not data.text.strip():
        return {"error": "Please enter some text"}

    prediction, explanation = analyze_with_llm(data.text)

    return {
        "prediction": prediction,
        "explanation": explanation
    }


# -----------------------------
# UI Route
# -----------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    with open("templates/index.html", "r") as f:
        return f.read()
