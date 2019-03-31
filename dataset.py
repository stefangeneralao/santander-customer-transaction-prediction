import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print('---> Fetching training data...')
train = pd.read_csv('train.csv')

print('---> Fetching test data...')
test = pd.read_csv('test.csv')

print('---> Processing dataset...')
X = train.iloc[:, 2:].values
y = train.iloc[:, 1].values
X_test = test.iloc[:, 1:].values

print('---> Scaling dataset...')
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

print('---> Splitting dataset...')
validation_split_ratio = 0.1
validation_split_index = int((1 - validation_split_ratio) * len(X))
X_train = X[:validation_split_index]
X_val = X[validation_split_index:]
y_train = y[:validation_split_index]
y_val = y[validation_split_index:]