import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from collections import Counter
import re
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv('IMDB Dataset.csv')

df.replace({'sentiment':{'positive':1, 'negative':0}}, inplace=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = ' '.join(text.split())
    return text

def build_vocab(texts):
    all_words = []
    for text in texts:
        text = preprocess_text(text)
        words = text.split()
        all_words.extend(words)
    word_counts = Counter(all_words)
    vocab_words = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common(10000-2)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
    return word_to_idx

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

class IMDBDatset(Dataset):
    def __init__(self, sequence, labels):
        self.sequence = sequence
        self.labels = labels
    
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequence[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)
    
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(list(sequences), batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sequences_padded, labels

vocab = build_vocab(df['review'].tolist())
sequences = []
for text in df['review']:
    seq = text_to_sequence(text, vocab)
    sequences.append(seq)
labels = df['sentiment'].tolist()
X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.3, stratify=labels,random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

trainset = IMDBDatset(X_train, y_train)
valset = IMDBDatset(X_val, y_val)
testset = IMDBDatset(X_test, y_test)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=collate_fn)
valloader = DataLoader(valset, batch_size=64, shuffle=False, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn=collate_fn)

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
    
model = SentimentLSTM(vocab_size=len(vocab)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for sequences, labels in trainloader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(trainloader)}")
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in valloader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Val Loss: {val_loss/len(valloader)}, Val Accuracy: {correct/total}")
        scheduler.step(val_loss)

torch.save(model.state_dict(), 'sentiment_lstm.pth')

model.load_state_dict(torch.load('sentiment_lstm.pth'))

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sequences, labels in testloader:
        sequences, labels = sequences.to(device), labels.to(device)

        outputs = model(sequences)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {correct/total}")
    
def predict(text, model, vocab):
    model.eval()
    sequence = text_to_sequence(text, vocab)
    sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sequence_tensor)
        if(output.item() >= 0.5):
            print("Positive")
        else:
            print("Negative")