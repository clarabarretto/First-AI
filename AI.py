import pandas as pd

dataset = pd.read_csv('cancer.csv')

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"]) # reading all columns besides the diagnosis column, input layer
y = dataset["diagnosis(1=m, 0=b)"] #only reads the diagnosis column

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import tensorflow as tf

model = tf.keras.models.Sequential()

# output = units, in this case = 256 neurons
# sigmoid function = plotting all the values from de network between 0 and 1

model.add(tf.keras.Input(shape=(x_train.shape[1],))) 
model.add(tf.keras.layers.Dense(256, activation='sigmoid')) 
model.add(tf.keras.layers.Dense(256, activation='sigmoid')) 
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #how the wheights are being tuned to fit the data

model.fit(x_train, y_train, epochs=1000) # how many times it is iterating over the same data

model.evaluate(x_test, y_test) # using new data this is the accuracy
