import openmeteo_requests

import requests_cache
import pandas as pd
import keras_tuner as kt
import os

from retry_requests import retry

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"

coor = [
    55.1226, 6.48353,
    54.173701, 6.25818,
    54.57597, 13.06216
]
    

hourly_dataframe = pd.DataFrame()

if os.path.exists("DataFrames/Off_Shore/Wind_Off_Shore_2018_2022.csv"):
    
    hourly_dataframe = pd.read_csv("DataFrames/Off_Shore/Wind_Off_Shore_2018_2022.csv")
    hourly_dataframe = hourly_dataframe.drop(hourly_dataframe.columns[[0]], axis=1)
    
else:

    hourly_dataframe = pd.DataFrame()

    for i in range(0, 3):
        
        params = {
            "latitude": coor[i * 2],
            "longitude": coor[i * 2 + 1],
            "start_date": "2018-01-01",
            "end_date": "2022-12-31",
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain", "snowfall",
                    "pressure_msl", "surface_pressure", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
                    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m"],
            "wind_speed_unit": "ms",
            "timezone": "Europe/Berlin"
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
        hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
        hourly_rain = hourly.Variables(4).ValuesAsNumpy()
        hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
        hourly_pressure_msl = hourly.Variables(6).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(7).ValuesAsNumpy()
        hourly_vapour_pressure_deficit = hourly.Variables(8).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(9).ValuesAsNumpy()
        hourly_wind_speed_100m = hourly.Variables(10).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(11).ValuesAsNumpy()
        hourly_wind_direction_100m = hourly.Variables(12).ValuesAsNumpy()
        hourly_wind_gusts_10m = hourly.Variables(13).ValuesAsNumpy()

        hourly_data = {}

        hourly_data["temperature_2m_" + str(i)] = hourly_temperature_2m
        hourly_data["relative_humidity_2m_" + str(i)] = hourly_relative_humidity_2m
        hourly_data["dew_point_2m_" + str(i)] = hourly_dew_point_2m
        hourly_data["apparent_temperature_" + str(i)] = hourly_apparent_temperature
        hourly_data["rain_" + str(i)] = hourly_rain
        hourly_data["snowfall_" + str(i)] = hourly_snowfall
        hourly_data["pressure_msl_" + str(i)] = hourly_pressure_msl
        hourly_data["surface_pressure_" + str(i)] = hourly_surface_pressure
        hourly_data["vapour_pressure_deficit_" + str(i)] = hourly_vapour_pressure_deficit
        hourly_data["wind_speed_10m_" + str(i)] = hourly_wind_speed_10m
        hourly_data["wind_speed_100m_" + str(i)] = hourly_wind_speed_100m
        hourly_data["wind_direction_10m_" + str(i)] = hourly_wind_direction_10m
        hourly_data["wind_direction_100m_" + str(i)] = hourly_wind_direction_100m
        hourly_data["wind_gusts_10m_" + str(i)] = hourly_wind_gusts_10m

        add_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe = pd.concat([hourly_dataframe, add_dataframe], axis=1)
        print(hourly_dataframe)
        

    print(hourly_dataframe)
    hourly_dataframe.to_csv("DataFrames/Off_Shore/Wind_Off_Shore_2018_2022.csv")

    # Add Colum from Off_Shore Power
    hourly_dataframe['Off_Shore'] = pd.read_csv("Power_Data/Wind_Off_Shore_2018_2022.csv")
    hourly_dataframe.to_csv("DataFrames/Off_Shore/Wind_Off_Shore_&_Wetter.csv")


print(hourly_dataframe)

# Split Data into useable Datasets

x = hourly_dataframe.drop(columns = "Off_Shore")
y = hourly_dataframe["Off_Shore"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=False)


# Normalize Data
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

  model.add(InputLayer((42,)))
  model.add(Dense(units=hp_layer_1, activation='relu'))
  model.add(Dense(units=hp_layer_2, activation='relu'))
  model.add(Dense(units=hp_layer_3, activation='relu'))
  model.add(Dense(1, 'linear'))

  model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                loss='mse',
                metrics=['mse'])
  
  return model


model_name = 'Off_Shore_model'

tuner = kt.Hyperband(model_builder,
                     objective="val_mse",
                     max_epochs=10,
                     factor=3,
                     directory='hyper_var_fitting/Off_Shore',
                     project_name=model_name)


stop_early = EarlyStopping(monitor='val_loss', patience=3)
tuner.search(x_train_preprocessed, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


model = Model

if os.path.exists("models/Off_Shore/" + model_name):
    model = load_model("models/Off_Shore/" + model_name)
    
else:
    cp = ModelCheckpoint("models/Off_Shore/" + model_name, save_best_only=True)
    model = tuner.hypermodel.build(best_hps)
    model.fit(x=x_train_preprocessed, y=y_train, validation_data=(x_test_preprocessed, y_test), callbacks=[stop_early, cp], epochs=100)


y_pred_test = model.predict(x_test_preprocessed)


plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Off_Shore Power")
plt.ylabel("Predicted Off_Shore Power")
plt.show()


result = pd.DataFrame(y_test)
result['Pred_Off_Shore'] = y_pred_test

mse = mean_squared_error(y_test, y_pred_test, squared=False)
print("MSE: ", mse)

plt.plot(result)
plt.legend(['Off_Shore', 'Pred_Off_Shore'])
plt.title("Mean_squared_error: " + mse.astype('str'))
plt.show()


timestamp_range = pd.date_range(start="2018-01-01", 
                                end="2022-12-31 23:45", freq="H")


plt.plot(timestamp_range[:200], y_test[:200])
plt.plot(timestamp_range[:200], result[:200])
plt.legend(['Validation', 'Val Predictions'])
plt.show()


#######################################
# 2023 TEST


dataframe_2023 = pd.DataFrame()


if os.path.exists("DataFrames/Off_Shore/Wind_Off_Shore_2023.csv"):
    
    dataframe_2023 = pd.read_csv("DataFrames/Off_Shore/Wind_Off_Shore_2023.csv")
    dataframe_2023 = dataframe_2023.drop(dataframe_2023.columns[[0]], axis=1)
    
else:

    for i in range(0, 3):
            
        params = {
            "latitude": coor[i * 2],
            "longitude": coor[i * 2 + 1],
            "start_date": "2023-01-01",
            "end_date": "2023-11-30",
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain", "snowfall",
                    "pressure_msl", "surface_pressure", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
                    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m"],
            "wind_speed_unit": "ms",
            "timezone": "Europe/Berlin"
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
        hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
        hourly_rain = hourly.Variables(4).ValuesAsNumpy()
        hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
        hourly_pressure_msl = hourly.Variables(6).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(7).ValuesAsNumpy()
        hourly_vapour_pressure_deficit = hourly.Variables(8).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(9).ValuesAsNumpy()
        hourly_wind_speed_100m = hourly.Variables(10).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(11).ValuesAsNumpy()
        hourly_wind_direction_100m = hourly.Variables(12).ValuesAsNumpy()
        hourly_wind_gusts_10m = hourly.Variables(13).ValuesAsNumpy()

        hourly_data = {}

        hourly_data["temperature_2m_" + str(i)] = hourly_temperature_2m
        hourly_data["relative_humidity_2m_" + str(i)] = hourly_relative_humidity_2m
        hourly_data["dew_point_2m_" + str(i)] = hourly_dew_point_2m
        hourly_data["apparent_temperature_" + str(i)] = hourly_apparent_temperature
        hourly_data["rain_" + str(i)] = hourly_rain
        hourly_data["snowfall_" + str(i)] = hourly_snowfall
        hourly_data["pressure_msl_" + str(i)] = hourly_pressure_msl
        hourly_data["surface_pressure_" + str(i)] = hourly_surface_pressure
        hourly_data["vapour_pressure_deficit_" + str(i)] = hourly_vapour_pressure_deficit
        hourly_data["wind_speed_10m_" + str(i)] = hourly_wind_speed_10m
        hourly_data["wind_speed_100m_" + str(i)] = hourly_wind_speed_100m
        hourly_data["wind_direction_10m_" + str(i)] = hourly_wind_direction_10m
        hourly_data["wind_direction_100m_" + str(i)] = hourly_wind_direction_100m
        hourly_data["wind_gusts_10m_" + str(i)] = hourly_wind_gusts_10m

        add_dataframe = pd.DataFrame(data = hourly_data)
        dataframe_2023 = pd.concat([dataframe_2023, add_dataframe], axis=1)
            
    dataframe_2023.to_csv("DataFrames/Off_Shore/Wind_Off_Shore_2023.csv")
        


# Normalize Data
scaler_train = StandardScaler().fit(dataframe_2023)
dataframe_2023_preprocessed = scaler_train.transform(dataframe_2023)


pred_2023 = model.predict(dataframe_2023_preprocessed)
off_shore_2023 = pd.read_csv("Power_Data/Wind_Off_Shore_2023.csv")


timestamp_range_2023 = pd.date_range(start="2023-01-01", 
                                end="2023-11-30 23:00", freq="H")

print(timestamp_range_2023)
timestamp_range_as_np_2023 = timestamp_range_2023.to_numpy()

mse_2023 = mean_squared_error(y_test, y_pred_test, squared=False)

plt.title("Wind Off shore RMSE 2023: " + str(mse_2023))
plt.plot(timestamp_range_as_np_2023[:500], off_shore_2023[:500])
plt.plot(timestamp_range_as_np_2023[:500], pred_2023[:500])
plt.legend(['off_shore_2023', 'pred_2023'])
plt.show()