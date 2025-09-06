# Movie Review Sentiment Analysis 🎬

A deep learning project that classifies movie reviews as positive or negative using a bidirectional LSTM neural network trained on the IMDB dataset. The project includes both training scripts and a user-friendly Streamlit web interface.

## 📋 Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)

## ✨ Features

- **Deep Learning Model**: Bidirectional LSTM with attention mechanism for accurate sentiment classification
- **High Accuracy**: Trained on 50,000 IMDB movie reviews with robust preprocessing
- **Web Interface**: Interactive Streamlit app for real-time sentiment analysis
- **Confidence Scoring**: Provides confidence scores for predictions
- **GPU Support**: Automatic GPU detection and utilization when available
- **Text Preprocessing**: Comprehensive text cleaning and normalization
- **Vocabulary Management**: Efficient vocabulary building with OOV handling

## 🚀 Demo

The web interface allows users to:
- Input movie reviews of any length
- Get instant sentiment predictions (Positive/Negative)
- View confidence scores for each prediction
- Process multiple reviews in real-time

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-sentiment-analysis.git
   cd movie-sentiment-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the IMDB dataset**
   - Download `IMDB Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Place it in the project root directory

## 🎯 Usage

### Training the Model

To train the model from scratch:

```bash
python train.py
```

This will:
- Load and preprocess the IMDB dataset
- Build vocabulary from the training data
- Train a bidirectional LSTM model
- Save the trained model as `sentiment_lstm.pth`
- Save the vocabulary as `vocab.pkl`

### Running the Web Interface


To launch the Streamlit web app:

```bash
streamlit run web.py
```

The app will be available at `http://localhost:8501`

## 🏗️ Model Architecture

The sentiment analysis model uses a **Bidirectional LSTM** architecture:

```
Input Text → Preprocessing → Embedding Layer → Bidirectional LSTM → Dropout → Dense Layer → Sigmoid → Prediction
```

### Key Components:
- **Embedding Layer**: 128-dimensional word embeddings
- **Bidirectional LSTM**: 2 layers with 256 hidden units each
- **Dropout**: 50% dropout for regularization
- **Output Layer**: Single neuron with sigmoid activation
- **Vocabulary Size**: 10,000 most frequent words

### Hyperparameters:
- Learning Rate: 0.001 (with ReduceLROnPlateau scheduler)
- Batch Size: 64
- Max Sequence Length: 500 tokens
- Epochs: 5
- Optimizer: Adam

## 📊 Dataset

- **Source**: IMDB Movie Reviews Dataset
- **Size**: 50,000 reviews (25,000 positive, 25,000 negative)
- **Split**: 70% training, 15% validation, 15% testing
- **Preprocessing**: 
  - HTML tag removal
  - Lowercase conversion
  - Special character removal
  - Sequence truncation/padding

## 🔧 Configuration

Key parameters that can be modified in `train.py`:

- `vocab_size`: Size of vocabulary (default: 10,000)
- `embedding_dim`: Embedding dimensions (default: 128)
- `hidden_dim`: LSTM hidden dimensions (default: 256)
- `num_layers`: Number of LSTM layers (default: 2)
- `dropout`: Dropout rate (default: 0.5)
- `batch_size`: Training batch size (default: 64)
- `epochs`: Training epochs (default: 5)


 ## 🙏 Acknowledgments

- IMDB for providing the movie reviews dataset
- PyTorch team for the deep learning framework
- Streamlit team for the web app framework
- The open-source community for various tools and libraries

---
