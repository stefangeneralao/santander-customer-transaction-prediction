import tensorflow as tf
from tensorflow.keras import layers

print('---> Building model...')
model = tf.keras.Sequential()

model.add(layers.Dense(1024, input_shape=(200,)))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(rate=0.5))

model.add(layers.Dense(1024))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(rate=0.5))

model.add(layers.Dense(512))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(rate=0.5))

model.add(layers.Dense(512))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(rate=0.5))

model.add(layers.Dense(128))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(rate=0.2))

model.add(layers.Dense(64))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(rate=0.2))

model.add(layers.Dense(8))
model.add(layers.LeakyReLU())

model.add(layers.Dense(8))
model.add(layers.LeakyReLU())

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
  optimizer=tf.optimizers.Adam(learning_rate=0.0003),
  loss='binary_crossentropy',
  metrics=['accuracy']
)