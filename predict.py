import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    prob = model.predict_proba(text_vec)

    if prediction[0] == 0:
        return "Fake", prob[0][0]
    else:
        return "Real", prob[0][1]

# Test input
news = input("Enter news text: ")

result, confidence = predict_news(news)

print(f"Prediction: {result}")
print(f"Confidence: {confidence:.2f}")