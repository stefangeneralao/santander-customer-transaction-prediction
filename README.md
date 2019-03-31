This repository is my attempt to crack the _Santander Customer Transaction Prediction_ at Kaggle, see https://kaggle.com/c/santander-customer-transaction-prediction .

An artificial feed-forward neural network was used as the model.

Almost 92% accuracy was achieved with the trained model. However, it was unfortunately not enough to win the competition.

## Download dataset
Download the training dataset from the following link: https://www.kaggle.com/c/santander-customer-transaction-prediction/download/train.csv

For generating a submission, the test dataset is required. Download the test dataset from the following link: https://www.kaggle.com/c/santander-customer-transaction-prediction/download/test.csv

Unpack the downloaded files. Move `train.csv` and `test.csv` to the root directory.

## Execution
Run `python3 main.py` to execute the analysis. At the end of the session the model and its corresponding weights will be saved to the `model`-directory.
