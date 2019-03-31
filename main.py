from model import model
from dataset import X_train, X_val, X_test, y_train, y_val
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import save_model, load_model

print('---> Fitting model...')
model.fit(
  X_train,
  y_train,
  validation_data=(X_val, y_val),
  shuffle=True,
  epochs=100,
  batch_size=128
)

print('---> Evaluating model...')
evaluation_loss = round(model.evaluate(X_val, y_val)[0], 5)

print('---> Predicting test...')
prediction = model.predict(X_test).flatten()

print('---> Exporting CSV...')
col1 = []
col2 = []
for i in range(prediction.size):
  col1.append('test_' + str(i))
  col2.append(prediction[i])

df = pd.DataFrame(data={
  'ID_code': col1, 'target': col2
})
date = datetime.datetime.now().strftime('%y-%m-%d-%H:%M:%S')
submission_filename = './submissions/submission_' + date + '_val-loss:' + str(evaluation_loss) + '.csv'
df.to_csv(submission_filename, sep=',',index=False)

print('---> Exporting model and weights...')
model_filename = './models/model_val-loss:' + str(evaluation_loss) + '.h5'
weights_filename = './models/weights_val-loss:' + str(evaluation_loss) + '.h5'
model.save(model_filename, include_optimizer=True)
model.save_weights(weights_filename)

print('---> Done.')