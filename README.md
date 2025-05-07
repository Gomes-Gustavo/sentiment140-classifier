# Sentiment140 Classifier

This repository contains a **sentiment analysis model** trained on the **Sentiment140 dataset**, using a custom **LSTM network with Word2Vec embeddings**. The project includes data exploration, preprocessing, training, and evaluation in a clear and reproducible format.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Model Performance](#model-performance)
- [References](#references)
- [Author](#author)

## Project Overview

This project builds a text classification pipeline that processes and classifies tweets into positive or negative sentiments using pre-trained Word2Vec embeddings and a two-layer LSTM model.

## Dataset

The **Sentiment140 dataset** contains **1.6 million tweets** annotated as:

- `0` = Negative sentiment
- `4` = Positive sentiment (converted to `1` in this project)

Each tweet includes metadata, but only the `text` and `sentiment` fields are used.

The dataset is available here: https://www.kaggle.com/kazanova/sentiment140

## Project Structure

```
sentiment140-classifier/
│
├── data/
│   ├── raw/                            # Raw Sentiment140 dataset
|       └── training.1600000.processed.noemoticon.csv 
│   ├── processed/                      # Cleaned and preprocessed versions
│   │   ├── sentiment140_clean.csv
│   │   ├── sentiment140_tokenized.csv
│   │   └── sentiment140_vectors.csv
│
├── models/
│   ├── lstm_sentiment140.h5           # Trained LSTM model
|   ├── word2vec_model                 # Gensim Word2Vec model
|   ├── word2vec_model.syn1neg.npy     # Word2Vec weights
|   └── word2vec_model.wv.vectors.npy  # Word2Vec vectors
|   
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Text cleaning, lemmatization, tokenization
│   ├── 03_word2vec_training.ipynb     # Word2Vec training + tweet vectorization
│   ├── 04_lstm_training_colab.ipynb   # LSTM model training (Google Colab)
│   └── 05_model_evaluation.ipynb      # Performance analysis and visualizations
│
├── requirements.txt                   # Required dependencies
├── README.md                          # Project documentation
└── .gitignore
```

## Jupyter Notebooks

| Step                             | Notebook                                                    | Description                                                                 |
| -------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------- |
| **1. Exploratory Data Analysis** | [01_eda.ipynb](notebooks/01_eda.ipynb)                      | Analyzes tweet length, missing values, sentiment distribution, etc.         |
| **2. Text Preprocessing**        | [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb) | Tokenization, lemmatization, stopword removal, cleaning.                    |
| **3. Word2Vec Embeddings**       | [03_word2vec_training.ipynb](notebooks/03_word2vec_training.ipynb)        | Trains Word2Vec and saves tweet embeddings as vectors.                      |
| **4. LSTM Model Training**       | [04_lstm_training_colab.ipynb](notebooks/04_lstm_training_colab.ipynb)  | Defines and trains the LSTM model with Dropout and EarlyStopping.           |
| **5. Evaluation & Visualization**| [05_model_evaluation.ipynb](notebooks/05_model_evaluation.ipynb) | Confusion matrix, classification report, predicted class distribution.      |

## Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/Gomes-Gustavo/sentiment140-classifier.git
cd sentiment140-classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Training the Model

To train the LSTM model, use the Google Colab notebook:

```python
# Open and run notebook 04_lstm_training.ipynb in Google Colab
```

After training, the model is saved to your Drive.

## Model Performance

| Metric               | Value                        |
|----------------------|------------------------------|
| **Validation Accuracy** | ~77.7%                    |
| **Embedding**           | Word2Vec                  |
| **Model**               | 2-layer LSTM with Dropout |
| **Dataset Size**        | 1.6 million tweets        |

The model performs well for a baseline sentiment classification task using pre-trained embeddings and basic tuning.

## Validation vs. Test Set

This project does not use a separate test set. The model was evaluated using a **hold-out validation set (20%)**, and the final accuracy (~77.7%) was derived from this validation data.

This decision was made for two main reasons:

- The **Sentiment140 dataset** already includes **1.6 million tweets**, and splitting it into training and validation provides a large and reliable sample for evaluation.
- The focus is to build a strong baseline model with proper training and validation, rather than deploying a production-ready system.

> If needed, the current project structure allows for easy inclusion of an external test set in the future — simply reserve a portion of the data or load an additional dataset.

## References

- [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/kazanova/sentiment140)

## Author

Developed by Gustavo Gomes

- [LinkedIn](https://www.linkedin.com/in/gustavo-gomes-581975333/)
