import pandas as pd

import keras_tuner as kt
import os

from retry_requests import retry

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam



    
# GET DATA
hourly_dataframe = pd.DataFrame()
   
hourly_dataframe = pd.read_csv("Power_Data/Price_2015_2022.csv")

# SPLIT DATA
x = hourly_dataframe.drop(columns = "Price")
y = hourly_dataframe["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=False)


# STANDARDIZATION
scaler_train = StandardScaler().fit(x_train)
x_train_preprocessed = scaler_train.transform(x_train)


scaler_test = StandardScaler().fit(x_test)
x_test_preprocessed = scaler_test.transform(x_test)



# AUTOMATIC HYPERPARAMETER TUNING
def model_builder(hp):
  model = Sequential()
 
  hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=1000, step=100)
  hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=1000, step=100)
  hp_layer_3 = hp.Int('layer_3', min_value=1, max_value=1000, step=100)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.add(InputLayer((6,)))
  model.add(Dense(units=hp_layer_1, activation='relu'))
  model.add(Dense(units=hp_layer_2, activation='relu'))
  model.add(Dense(units=hp_layer_3, activation='relu'))
  model.add(Dense(1, 'linear'))

  model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                loss='mse',
                metrics=['mse'])
  
  return model


model_name = 'Price_model'

tuner = kt.Hyperband(model_builder,
                     objective="val_mse",
                     max_epochs=10,
                     factor=3,
                     directory='hyper_var_fitting/Price',
                     project_name=model_name)


stop_early = EarlyStopping(monitor='val_loss', patience=3)
tuner.search(x_train_preprocessed, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


model = Model

if os.path.exists("models/Price/" + model_name):
    model = load_model("models/Price/" + model_name)
    
else:
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    cp = ModelCheckpoint("models/Price/" + model_name, save_best_only=True)
    model = tuner.hypermodel.build(best_hps)
    model.fit(x=x_train_preprocessed, y=y_train, validation_data=(x_test_preprocessed, y_test), callbacks=[early_stopping, cp], epochs=100)


y_pred_test = model.predict(x_test_preprocessed)


# TEST
plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Price Power")
plt.ylabel("Predicted Price Power")
plt.show()


timestamp_range = pd.date_range(start="2015-01-01", 
                                end="2022-12-31 23:45", freq="H")

plt.plot(timestamp_range[:200], y_test[200:400])
plt.plot(timestamp_range[:200], y_pred_test[200:400])
plt.legend(['Validation', 'Val Predictions'])
plt.show()

