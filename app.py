from flask import Flask, render_template, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

app = Flask(__name__)

# Load the sentiment model and tokenizer
sentiment_model_path = 'sentiment_model'
tokenizer = DistilBertTokenizer.from_pretrained(sentiment_model_path)
model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the emotion detection pipeline from Hugging Face
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Function to predict sentiment
def predict_sentiment(input_text):
    # Tokenize the input text
    encoding = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoding = {key: val.to(device) for key, val in encoding.items()}
    
    # Run the model
    with torch.no_grad():
        outputs = model(**encoding)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
    
    # Ensure the output is in the expected format (3 classes)
    if len(probabilities) != 3:
        raise ValueError("Unexpected number of output classes. Expected 3 probabilities.")
    
    # Convert probabilities to regular Python floats
    probabilities = [float(prob) for prob in probabilities]
    
    # Get the predicted class index and sentiment label
    predicted_class = torch.argmax(torch.tensor(probabilities), dim=-1).item()
    sentiment = ['Negative', 'Neutral', 'Positive'][predicted_class]
    
    return sentiment, probabilities

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results.html')
def analysis_page():
    return render_template('results.html')

@app.route('/analyze')
def analyze():
    text = request.args.get('text')
    
    try:
        # Predict sentiment
        sentiment, probabilities = predict_sentiment(text)
        
        # Predict emotion
        emotion_result = emotion_pipeline(text)
        emotion = emotion_result[0]['label']  # Get the most probable emotion label
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    # Send sentiment, probabilities, and emotion as JSON
    return jsonify({
        'sentiment': sentiment,
        'probabilities': {
            'negative': probabilities[0],
            'neutral': probabilities[1],
            'positive': probabilities[2]
        },
        'emotion': emotion
    })

if __name__ == "__main__":
    app.run(debug=True)
