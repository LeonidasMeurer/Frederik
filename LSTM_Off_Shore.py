from copy import deepcopy

import pandas as pd
import numpy as np
import os


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError



hourly_dataframe = pd.read_csv("Power_Data/Off_Shore_2018_2022_ViertelStunde.csv")


print(hourly_dataframe)

timestamp_range = pd.date_range(start="2018-01-01", 
                                end="2022-12-31 23:45", freq="15min")
print(timestamp_range)



# Split Data into useable Datasets
def df_to_X_y(df, window_size):
  df_as_np = df.to_numpy()
  x = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [a for a in df_as_np[i:i+window_size]]
    x.append(row)

    label = df_as_np[i+window_size]
    y.append(label[0]) #[0]
  return np.array(x), np.array(y)

WINDOW_SIZE = 5
x, y = df_to_X_y(hourly_dataframe, WINDOW_SIZE)



timestamp_range_as_np = timestamp_range.to_numpy()
dates_train, x_train, y_train = timestamp_range_as_np[:130000], x[:130000], y[:130000]
dates_val, x_val, y_val = timestamp_range_as_np[130000:150000], x[130000:150000], y[130000:150000]
dates_test, x_test, y_test = timestamp_range_as_np[150000:], x[150000:], y[150000:]


 
model = Sequential()
model.add(InputLayer((96, 1)))
model.add(GRU(64))
model.add(Dense(8, 'relu'))
model.add(Dense(1, 'linear'))
model.summary()

cp = ModelCheckpoint('models/LSTM/Off_Shore_24_Std', save_best_only=True)
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])



model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[cp])

model = load_model("models/LSTM/Off_Shore")

val_predictions = model.predict(x_test).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_test})

plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])
plt.show()

test_predictions = model.predict(x_val).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_val})

plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])
plt.show()



train_predictions = model.predict(x_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
print(train_results)


recursive_predictions = []
recursive_dates = timestamp_range_as_np[130000:130100]
print(recursive_dates)

last_window = deepcopy(x_train[-1])

for target_date in range(0, 100):
  next_prediction = model.predict(np.array([last_window]), verbose=0).flatten()
  recursive_predictions.append(next_prediction)
  last_window = np.roll(last_window, -1)
  last_window[-1] = next_prediction
  
print(train_results['Actuals'][129800:])
print(val_results['Actuals'][:200])
print(hourly_dataframe[129995:130005])

plt.plot(dates_val[:100], recursive_predictions)
plt.plot(dates_train[129800:], train_results['Actuals'][129800:])
plt.plot(dates_train[129800:], train_results['Train Predictions'][129800:])
plt.plot(dates_val[:200], val_results['Actuals'][:200])
plt.plot(dates_val[:200], val_results['Val Predictions'][:200])
plt.plot(timestamp_range_as_np[129900:130100], hourly_dataframe[129900:130100])
plt.legend(['Recursive Predictions', 'Train Actual', 'Train Predictions', 'Validation', 'Val Predictions','train'])
plt.show()