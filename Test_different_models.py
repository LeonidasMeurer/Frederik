import openmeteo_requests

import requests_cache
import pandas as pd
import numpy as np
import keras_tuner as kt

from retry_requests import retry

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model
from keras.losses import SparseCategoricalCrossentropy


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 55.1226,
	"longitude": 6.48353,
	"start_date": "2022-01-01",
	"end_date": "2022-12-31",
	"hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m"],
	"wind_speed_unit": "ms"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
hourly_wind_speed_100m = hourly.Variables(3).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(4).ValuesAsNumpy()
hourly_wind_direction_100m = hourly.Variables(5).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(6).ValuesAsNumpy()

hourly_data = {}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_speed_120m"] = hourly_wind_speed_100m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_direction_120m"] = hourly_wind_direction_100m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

hourly_dataframe = pd.DataFrame(data = hourly_data)
print(hourly_dataframe)


# Add Colum from Off_Shore Power
hourly_dataframe['Off_Shore'] = pd.read_csv("Power_Data\Off_Shore_2022.csv")
 

# Split Data into useable Datasets

x = hourly_dataframe.drop(columns = "Off_Shore")
y = hourly_dataframe["Off_Shore"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# Normalize Data

scaler = StandardScaler().fit(x_train)

def preproccesor(X):
    A = np.copy(X)
    A = scaler.transform(A)
    return A

x_train_preprocessed, x_test_preprocessed = preproccesor(x_train), preproccesor(x_test)

# LINEAR REGRESSION
# Fit Data

lr = LinearRegression().fit(x_train_preprocessed, y_train)

# Run Training

y_pred_train = lr.predict(x_train_preprocessed)
y_pred_test = lr.predict(x_test_preprocessed)


# Compare MSE
print("===============================")
print("LINEAR REGRESSION")
print("mean_squared_error train", mean_squared_error(y_train, y_pred_train, squared=False))
print("mean_squared_error test", mean_squared_error(y_test, y_pred_test, squared=False))


# mean_squared_error train 1151.668452492577
# mean_squared_error test 1164.2652576338912



# NEIGHBOURS REGRESSOR

knn = KNeighborsRegressor(n_neighbors=7).fit(x_train_preprocessed, y_train)

y_pred_train = knn.predict(x_train_preprocessed)
y_pred_test = knn.predict(x_test_preprocessed)

print("===============================")
print("NEIGHBOURS REGRESSOR")
print("mean_squared_error train", mean_squared_error(y_train, y_pred_train, squared=False))
print("mean_squared_error test", mean_squared_error(y_test, y_pred_test, squared=False))


# mean_squared_error train 760.2925108959595
# mean_squared_error test 893.3588231408805


# RANDOM FOREST

rfr = RandomForestRegressor(max_depth=14).fit(x_train_preprocessed, y_train)

y_pred_train = rfr.predict(x_train_preprocessed)
y_pred_test = rfr.predict(x_test_preprocessed)


# mean_squared_error train 458.7767472500959
# mean_squared_error test 883.6525318983564


print("===============================")
print("RANDOM FOREST")
print("mean_squared_error train", mean_squared_error(y_train, y_pred_train, squared=False))
print("mean_squared_error test", mean_squared_error(y_test, y_pred_test, squared=False))


# GRADIENT BOOSTING

gbr = GradientBoostingRegressor(n_estimators=400).fit(x_train_preprocessed, y_train)

y_pred_train = gbr.predict(x_train_preprocessed)
y_pred_test = gbr.predict(x_test_preprocessed)


# mean_squared_error train 758.4647996291867
# mean_squared_error test 927.908820106642


print("===============================")
print("GRADIENT BOOSTING")
print("mean_squared_error train", mean_squared_error(y_train, y_pred_train, squared=False))
print("mean_squared_error test", mean_squared_error(y_test, y_pred_test, squared=False))


# NEURAL NETWORK
# SIMPLE
simple_nn = Sequential()
simple_nn.add(InputLayer((7,)))
simple_nn.add(Dense(2, 'relu'))
simple_nn.add(Dense(1, 'linear'))

opt = Adam(learning_rate=0.1)
cp = ModelCheckpoint("models/Testing_Models/simple_nn", save_best_only=True)
simple_nn.compile(optimizer=opt, loss="mse", metrics=[RootMeanSquaredError()])
simple_nn.fit(x=x_train_preprocessed, y=y_train, validation_data=(x_test_preprocessed, y_test), callbacks=[cp], epochs=100)

# loss: 3427195.2500 - root_mean_squared_error: 1851.2686 - val_loss: 3347057.0000 - val_root_mean_squared_error: 1829.4963

y_pred_train = simple_nn.predict(x_train_preprocessed)
y_pred_test = simple_nn.predict(x_test_preprocessed)

print("===============================")
print("NEURAL NETWORK")
print("mean_squared_error train", mean_squared_error(y_train, y_pred_train, squared=False))
print("mean_squared_error test", mean_squared_error(y_test, y_pred_test, squared=False))


# MEDIUM
medium_nn = Sequential()
medium_nn.add(InputLayer((7,)))
medium_nn.add(Dense(32, 'relu'))
medium_nn.add(Dense(16, 'relu'))
medium_nn.add(Dense(1, 'linear'))

opt = Adam(learning_rate=0.1)
cp = ModelCheckpoint("models/Testing_Models/medium_nn", save_best_only=True)
medium_nn.compile(optimizer=opt, loss="mse", metrics=[RootMeanSquaredError()])
medium_nn.fit(x=x_train_preprocessed, y=y_train, validation_data=(x_test_preprocessed, y_test), callbacks=[cp], epochs=100)

# loss: 834542.3125 - root_mean_squared_error: 913.5329 - val_loss: 1007373.3125 - val_root_mean_squared_error: 1003.6799


# LARGE
large_nn = Sequential()
large_nn.add(InputLayer((7,)))
large_nn.add(Dense(256, 'relu'))
large_nn.add(Dense(128, 'relu'))
large_nn.add(Dense(64, 'relu'))
large_nn.add(Dense(32, 'relu'))
large_nn.add(Dense(16, 'relu'))
large_nn.add(Dense(1, 'linear'))

opt = Adam(learning_rate=0.1)
cp = ModelCheckpoint("models/Testing_Models/large_nn", save_best_only=True)
large_nn.compile(optimizer=opt, loss="mse", metrics=[RootMeanSquaredError()])
large_nn.fit(x=x_train_preprocessed, y=y_train, validation_data=(x_test_preprocessed, y_test), callbacks=[cp], epochs=100)

# loss: 961076.5000 - root_mean_squared_error: 980.3451 - val_loss: 997375.1875 - val_root_mean_squared_error: 998.6867



# AUTOMATIC HYPERPARAMETER TUNING
def model_builder(hp):
  model = Sequential()
 
  hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
  hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=1000, step=100)
  hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=1000, step=100)
  hp_layer_3 = hp.Int('layer_3', min_value=1, max_value=1000, step=100)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.add(InputLayer((7,)))
  model.add(Dense(units=hp_layer_1, activation=hp_activation))
  model.add(Dense(units=hp_layer_2, activation=hp_activation))
  model.add(Dense(units=hp_layer_3, activation=hp_activation))
  model.add(Dense(1, 'linear'))

  model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                loss='mse',
                metrics=['mse'])
  
  return model


tuner = kt.Hyperband(model_builder,
                     objective="val_mse",
                     max_epochs=10,
                     factor=3,
                     directory='hyper_var_fitting/Testing_Models',
                     project_name='hyperpar_auto_fitted')


stop_early = EarlyStopping(monitor='val_loss', patience=3)

tuner.search(x_train_preprocessed, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

cp = ModelCheckpoint("models/Testing_Models/hyperpar_auto_fitted", save_best_only=True)

model = tuner.hypermodel.build(best_hps)

model.fit(x=x_train_preprocessed, y=y_train, validation_data=(x_test_preprocessed, y_test), callbacks=[cp], epochs=100)

y_pred_train = model.predict(x_train_preprocessed)
y_pred_test = model.predict(x_test_preprocessed)

print("===============================")
print("NEURAL NETWORK")
print("mean_squared_error train", mean_squared_error(y_train, y_pred_train, squared=False))
print("mean_squared_error test", mean_squared_error(y_test, y_pred_test, squared=False))


plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Off_Shore Power")
plt.ylabel("Predicted Off_Shore Power")
plt.show()

plt.plot(y_pred_test) 
plt.show() 

# 2 LAYER
# mean_squared_error train 834.1267243098215
# mean_squared_error test 935.3657824811997

# 3 LAYER
# mean_squared_error train 797.9557090876757
# mean_squared_error test 912.8953003820801

# 4 Layer
# mean_squared_error train 773.3028648718483
# mean_squared_error test 913.8496301497748

# 5 Jahre mit 4 Layer
# mean_squared_error train 690.0060011057519
# mean_squared_error test 845.3391639737355

######################################
# 5 Jahre mit 4 Layer 4 Columns
# mean_squared_error train 884.5014017637324
# mean_squared_error test 909.1021724623415
