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

| Model                | Accuracy | Notes                          |
|----------------------|----------|--------------------------------|
| LSTM                 | 50.5%    | Single-layer implementation    |
| GRU (1-layer)        | 49.9%    | Base GRU architecture          |
| GRU (1-layer, 10-epoch)| 50.3% | Extended training duration     |
| GRU (2-layer)        | 50.5%    | Deeper architecture            |
| LSTM-GRU Hybrid      | 50.7%    | Combined approach              |

Key findings:
- All models achieved comparable performance (49.9%-50.7% accuracy)
- The hybrid LSTM-GRU showed marginal improvement (50.7%)
- ROC curve analysis revealed similar discriminative power across architectures

### Performance Analysis
1. **Key Observations**:
   - All models performed near random chance (50% baseline)
   - Hybrid architecture showed marginal improvement (+0.2% over best single model)
   - GRU variants underperformed despite theoretical efficiency advantages

   ### Accuracy Interpretation
The near-random performance (49.9%-50.7%) indicates:
- Models failed to learn meaningful patterns beyond random guessing
- Potential contributing factors:
  - **Data Challenges**: Unhandled negations, sarcasm, or ambiguous language
  - **Architecture Limitations**: Vanilla RNNs may struggle with long-range dependencies
  - **Training Dynamics**: Early stopping may have prevented convergence

### Error Analysis
Manual review of 100 misclassified samples revealed:

1. **Common Error Patterns**:
   ```text
   "Not even worth the free streaming" (Predicted: Positive | Actual: Negative)
   "So bad it circles back to fun" (Predicted: Negative | Actual: Positive)

## Implementation Details

The project includes a full NLP pipeline:
1. Text preprocessing (tokenization, vocabulary encoding)
2. Sequence padding and dataset batching
3. Model architecture development in TensorFlow/Keras
4. Training with early stopping and validation
5. Performance evaluation (accuracy, ROC curves, confusion matrices)

## Technologies Used

- **Core**: Python 3.7+, TensorFlow 2.x, Keras
- **Data Processing**: NumPy, pandas
- **Visualization**: Matplotlib, Seaborn
- **Environment Management**: pip, virtualenv

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/iankamar/sentiment-analysis-rnn.git
cd sentiment-analysis-rnn

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab

```

## License

This project code is licensed for personal and educational use.

The dataset used is the [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) available on Kaggle. The dataset is intended for academic and non-commercial use only. Please refer to the Kaggle page for specific licensing and usage terms.

