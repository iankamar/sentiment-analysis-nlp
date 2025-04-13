# Sentiment Analysis with RNNs

This repository contains Jupyter notebooks implementing various recurrent neural network (RNN) architectures for sentiment analysis on IMDb movie reviews.

## Project Overview

This project explores and compares different RNN-based models for binary text classification, including:
- Single-layer LSTM networks
- Single-layer GRU networks
- Multi-layer GRU networks
- Combined LSTM-GRU hybrid architectures

## Models and Results

The following models were implemented and evaluated:
- **LSTM**: Single-layer LSTM model — 68.5% accuracy
- **GRU (1-layer)**: Single-layer GRU model — 68.2% accuracy
- **GRU (2-layer)**: Two-layer GRU model — 32.2% accuracy
- **LSTM-GRU Hybrid**: Combined architecture — 63.3% accuracy

Performance was assessed using accuracy and ROC curve analysis. Single-layer models showed the most consistent results.

## Implementation Details

The project includes a full NLP pipeline:
- Text tokenization and vocabulary encoding
- Vocabulary construction and padding
- Dataset batching with TensorFlow
- RNN model construction and training
- Evaluation and visualization of performance metrics

## Technologies Used

- Python 3.7+
- TensorFlow / Keras
- NumPy
- pandas
- Matplotlib
- Seaborn

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/iankamar/sentiment-analysis-rnn.git

# Navigate to the project directory
cd sentiment-analysis-rnn

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt


## License

This project code is licensed for personal and educational use.

The dataset used is the [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) available on Kaggle. The dataset is intended for academic and non-commercial use only. Please refer to the Kaggle page for specific licensing and usage terms.

