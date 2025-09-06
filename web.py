import torch
from torch import nn
import streamlit as st
import re
import pickle

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.dropout(hidden)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output.squeeze(-1)
    
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def text_to_sequence(text, word_to_idx):
    clean_text = preprocess_text(text)
    words = clean_text.split()
    sequence = []

    for word in words:
        if word in word_to_idx:
            sequence.append(word_to_idx[word])
        else:
            sequence.append(word_to_idx['<UNK>'])
    
    if len(sequence) > 500:
        sequence = sequence[:500]
    
    return sequence

@st.cache_resource
def load_model_and_vocab():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentLSTM(vocab_size=len(vocab)).to(device)

    model.load_state_dict(torch.load('sentiment_lstm.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, vocab, device

model, vocab, device = load_model_and_vocab()

st.set_page_config(page_title="Movie Review Sentiment Analyser", layout="wide")
st.title("Movie Review Sentiment Analyser")
st.write("Enter a movie review below to analyze its sentiment.")

review_text = st.text_area("Enter your review here:", height=200)
if st.button("Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review before analyzing.")
    else:
        sequence = text_to_sequence(review_text, vocab)
        sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(sequence_tensor).item()
        
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        
        st.subheader("Sentiment Analysis Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")