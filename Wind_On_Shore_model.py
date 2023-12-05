import openmeteo_requests

import requests_cache
import pandas as pd
import numpy as np
import keras_tuner as kt
import time
import os

from retry_requests import retry

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


from keras.layers import *
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam



hourly_dataframe = pd.DataFrame()

coor = [
    49.328626795060124, 9.425528005537943,
    49.95173588376352, 7.410436294711973,
    50.520253073391906, 6.3973463623328914,
    51.0976495650322, 10.648546842504642,
    51.0912510418498, 14.65686802679879,
    51.86111406584052, 13.752624840052972,
    52.184640955506545, 11.78690833547562,
    53.174811926167976, 14.272126488402474,
    52.72468932664566, 11.905242606494644,
    52.85017765335744, 9.608533621730034,
    53.39655145215375, 7.362176866279748,
    53.908326804335644, 9.119930662349349,
    54.75157066690064, 8.866425769000823,
    54.47585186573757, 11.13224500144233,
    53.842776821762506, 12.097029172663582
]

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"

if os.path.exists("DataFrames/On_Shore/Wind_On_Shore_2015_2022_14Std_&_Wetter.csv"):
    
    hourly_dataframe = pd.read_csv("DataFrames/On_Shore/Wind_On_Shore_2015_2022_14Std_&_Wetter.csv")
    hourly_dataframe = hourly_dataframe.drop(hourly_dataframe.columns[[0]], axis=1)
    
else:
    
    hourly_dataframe = pd.DataFrame()

    for i in range(0, 15):
        
        params = {
            "latitude": coor[i * 2],
            "longitude": coor[i * 2 + 1],
            "start_date": "2015-01-01",
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

        hourly_data["temperature_2m_" + str(i)] = hourly_temperature_2m
        hourly_data["relative_humidity_2m_" + str(i)] = hourly_relative_humidity_2m
        hourly_data["wind_speed_10m_" + str(i)] = hourly_wind_speed_10m
        hourly_data["wind_speed_100m_" + str(i)] = hourly_wind_speed_100m
        hourly_data["wind_direction_10m_" + str(i)] = hourly_wind_direction_10m
        hourly_data["wind_direction_100m_" + str(i)] = hourly_wind_direction_100m
        hourly_data["wind_gusts_10m_" + str(i)] = hourly_wind_gusts_10m

        add_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe = pd.concat([hourly_dataframe, add_dataframe], axis=1)
        print(hourly_dataframe)

    print(hourly_dataframe)
    hourly_dataframe.to_csv("DataFrames/On_Shore/Wind_On_Shore_2015_2022_15Std.csv")

    # Add Colum from On_Shore Power
    hourly_dataframe['On_Shore'] = pd.read_csv("Power_Data/On_Shore_2015_2022.csv")
    hourly_dataframe.to_csv("DataFrames/On_Shore/Wind_On_Shore_2015_2022_14Std_&_Wetter.csv")


# Split Data into useable Datasets
x = hourly_dataframe.drop(columns = "On_Shore")
y = hourly_dataframe["On_Shore"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=False)


# Normalize Data
scaler = StandardScaler().fit(x_train)

def preproccesor(X):
    A = np.copy(X)
    A = scaler.transform(A)
    return A

x_train_preprocessed, x_test_preprocessed = preproccesor(x_train), preproccesor(x_test)

# AUTOMATIC HYPERPARAMETER TUNING
def model_builder(hp):
  model = Sequential()
 
  hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=1000, step=100)
  hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=1000, step=100)
  hp_layer_3 = hp.Int('layer_3', min_value=1, max_value=1000, step=100)
  hp_layer_4 = hp.Int('layer_4', min_value=1, max_value=1000, step=100)
  hp_layer_5 = hp.Int('layer_5', min_value=1, max_value=1000, step=100)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.add(InputLayer((105,)))
  model.add(Dense(units=hp_layer_1, activation='relu'))
  model.add(Dense(units=hp_layer_2, activation='relu'))
  model.add(Dense(units=hp_layer_3, activation='relu'))
  model.add(Dense(units=hp_layer_4, activation='relu'))
  model.add(Dense(units=hp_layer_5, activation='relu'))
  model.add(Dense(1, 'linear'))

  model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                loss='mse',
                metrics=['mse'])
  
  return model

model_name = 'autoF_2015_2022_14Stand_5Ly'

tuner = kt.Hyperband(model_builder,
                     objective="val_mse",
                     max_epochs=10,
                     factor=3,
                     directory='hyper_var_fitting/On_Shore',
                     project_name=model_name)


stop_early = EarlyStopping(monitor='val_loss', patience=3)
tuner.search(x_train_preprocessed, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = Model

if os.path.exists("models/On_Shore/" + model_name):
    model = load_model("models/On_Shore/" + model_name)
    
else:
    cp = ModelCheckpoint("models/On_Shore/" + model_name, save_best_only=True)
    model = tuner.hypermodel.build(best_hps)
    model.fit(x=x_train_preprocessed, y=y_train, validation_data=(x_test_preprocessed, y_test), callbacks=[cp], epochs=100)


y_pred_test = model.predict(x_test_preprocessed)


plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual On_Shore Power")
plt.ylabel("Predicted On_Shore Power")
plt.show()


result = pd.DataFrame(y_test)
result['Pred_On_Shore'] = y_pred_test

mse = mean_squared_error(y_test, y_pred_test, squared=False)
print("MSE: ", mse)

plt.plot(result)
plt.legend(['On_Shore', 'Pred_On_Shore'])
plt.title("Mean_squared_error: " + mse.astype('str'))
plt.show()

# autoF_2015_2022_14Stand
# MSE: 3089.0734695043834

# autoF_2015_2022_14Stand_5Ly
# MSE: 3082.261472353908


#######################################
# 2023 Bsp

dataframe_2023 = pd.DataFrame()

if os.path.exists("DataFrames/On_Shore/Wind_On_Shore_2023_bis_Nov.csv"):
    
    dataframe_2023 = pd.read_csv("DataFrames/On_Shore/Wind_On_Shore_2023_bis_Nov.csv")
    dataframe_2023 = dataframe_2023.drop(dataframe_2023.columns[[0]], axis=1)
    
else:
    
    for i in range(0, 15):
        
        params = {
            "latitude": coor[i * 2],
            "longitude": coor[i * 2 + 1],
            "start_date": "2023-01-01",
            "end_date": "2023-11-31",
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

        hourly_data["temperature_2m_" + str(i)] = hourly_temperature_2m
        hourly_data["relative_humidity_2m_" + str(i)] = hourly_relative_humidity_2m
        hourly_data["wind_speed_10m_" + str(i)] = hourly_wind_speed_10m
        hourly_data["wind_speed_100m_" + str(i)] = hourly_wind_speed_100m
        hourly_data["wind_direction_10m_" + str(i)] = hourly_wind_direction_10m
        hourly_data["wind_direction_100m_" + str(i)] = hourly_wind_direction_100m
        hourly_data["wind_gusts_10m_" + str(i)] = hourly_wind_gusts_10m

        add_dataframe = pd.DataFrame(data = hourly_data)
        dataframe_2023 = pd.concat([dataframe_2023, add_dataframe], axis=1)

    dataframe_2023.to_csv("DataFrames/On_Shore/Wind_On_Shore_2023_bis_Nov.csv")
    
    
dataframe_2023_preprocessed = preproccesor(dataframe_2023)

pred_2023 = model.predict(dataframe_2023_preprocessed)
off_shore_2023 = pd.read_csv("Power_Data/On_Shore_2023_bis_Nov.csv")


timestamp_range_2023 = pd.date_range(start="2023-01-01", 
                                end="2023-11-30 23:00", freq="H")

print(timestamp_range_2023)
timestamp_range_as_np_2023 = timestamp_range_2023.to_numpy()

plt.plot(timestamp_range_as_np_2023[:500], off_shore_2023[:500])
plt.plot(timestamp_range_as_np_2023[:500], pred_2023[:500])
plt.legend(['on_shore_2023', 'pred_2023'])
plt.show()