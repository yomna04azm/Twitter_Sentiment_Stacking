from flask import Flask, request, jsonify, render_template
import torch
import joblib
import pickle
from transformers import BertTokenizer, BertModel

model = joblib.load("sentiment_stack_model.pkl")

with open("bert_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

app = Flask(__name__)

def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy().reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    text = ""

    if request.method == 'POST':
        text = request.form['tweet']
        if text:
            emb = get_bert_embedding(text)
            pred = model.predict(emb)[0]
            sentiment = 'Positive' if pred == 1 else 'Negative'

    return render_template('index.html', sentiment=sentiment, text=text)

if __name__ == '__main__':
    app.run(debug=True)

