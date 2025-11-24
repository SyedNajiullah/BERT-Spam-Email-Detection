# BERT Spam Detection with Logistic Regression

This project demonstrates text classification (Spam vs. Ham) using BERT embeddings and Logistic Regression, implemented in a Jupyter Notebook. The workflow loads the SMS Spam Collection dataset, processes text through BERT, and trains a classifier achieving high accuracy.

## Features

- **Text classification using BERT embeddings**
- **SMS Spam Collection dataset (from Kaggle)**
- **Train/test split and evaluation**
- **Logistic Regression for classification**
- **Performance metrics: accuracy, precision, recall, f1-score**
- **Confusion matrix visualization with Seaborn**

## Dataset

Download the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it as `spam.csv` in the project directory.

## Requirements

Install required libraries before running the notebook:

```
pip install numpy pandas seaborn tensorflow scikit-learn transformers
```

## Usage

1. Clone this repository and open `BERT.ipynb` in Jupyter or Colab.
2. Ensure `spam.csv` is present in the working directory.
3. Run the cells step-by-step:
   - Import libraries
   - Load data
   - Process data
   - Extract BERT embeddings
   - Train and evaluate the classifier
   - Visualize confusion matrix

## Model

- Uses `TFBertModel` and `BertTokenizer` from HuggingFace Transformers
- Embeddings from the CLS token
- Logistic Regression via scikit-learn

## Results

- Achieves ~99% accuracy in spam detection based on the test set
