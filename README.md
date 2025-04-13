# Sentiment Analysis with RNNs

This repository contains Jupyter notebooks implementing various recurrent neural network architectures for sentiment analysis on IMDb movie reviews.

## Project Overview

This project explores and compares different RNN approaches for text classification, including:
- Single-layer LSTM networks
- Single-layer GRU networks
- Multi-layer GRU networks
- Combined LSTM-GRU architectures

## Models and Results

The following models were implemented and evaluated:
- **LSTM**: Single-layer LSTM model achieving 68.5% accuracy
- **GRU (1-layer)**: Single-layer GRU model achieving 68.2% accuracy
- **GRU (2-layer)**: Two-layer GRU model achieving 32.2% accuracy
- **LSTM-GRU Hybrid**: Combined architecture with 63.3% accuracy

Model performance was evaluated using both accuracy metrics and ROC curve analysis, with the single-layer models demonstrating the most consistent performance.

## Implementation Details

The project implements a complete NLP pipeline including:
- Text tokenization and encoding
- Vocabulary construction
- Sequence padding and batching
- Model training and evaluation
- Performance visualization

## Technologies Used

- TensorFlow/Keras
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/iankamar/sentiment-analysis-rnn.git

# Navigate to the directory
cd sentiment-analysis-rnn

# Create a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

## License
Copyright © 2024 Ian Kamar. All rights reserved.