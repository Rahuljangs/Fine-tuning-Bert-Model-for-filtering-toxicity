

# Toxic Comment Classification with BERT

This project utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for toxic comment classification. The model is trained to classify comments as toxic or non-toxic.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Dependencies](#dependencies)

## Introduction

Toxic comments are prevalent on many online platforms and can have harmful effects on users. This project aims to automatically classify comments as toxic or non-toxic using natural language processing (NLP) techniques, specifically leveraging the BERT model.

## Installation

Ensure you have the necessary dependencies installed. You can install them via pip:

```bash
pip install transformers
pip install pandas
pip install scikit-learn
pip install torch
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/toxic-comment-classification.git
cd toxic-comment-classification
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) and place it in the project directory.

3. Run the code:

```bash
python toxic_comment_classification.py
```

## Dataset

The dataset used for training is sourced from [Kaggle's Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data). It consists of comments labeled as toxic or non-toxic.

## Training

The model is trained using the BERT model architecture. Training is performed using PyTorch. The `Trainer` class from the `transformers` library is utilized for training.

## Evaluation

Evaluation metrics such as accuracy, precision, recall, and F1-score are computed to assess the model's performance. These metrics are calculated on a validation dataset.

## Inference

After training, the model can be used for inference on new comments. Provide the comment text as input, and the model will output the probability of it being toxic.

## Dependencies

- transformers
- pandas
- scikit-learn
- torch

