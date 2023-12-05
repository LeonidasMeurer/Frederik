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


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"

coor = [
    48.398328420837885, 13.30707631816199,
    48.07208291702251, 10.6884226850224,
    48.72363048878581, 12.43092644074699,
    49.610902961455494, 11.797797324428474,
    49.65024936369768, 9.937617015287776,
    50.86081546791219, 12.185222242809857,
    51.34398118614598, 14.947863172280492,
    51.37014773535029, 13.395400915099836,
    51.58120344941649, 11.48926323366893,
    52.39150245130204, 11.78590068037918,
    52.51819552310499, 13.080703341948448,
    52.66035823395796, 14.229744816655575,
    53.924543827740706, 13.21731608773746,
    53.58255526598762, 11.41196034636924,
    54.62834795675269, 9.355277496645371,
    53.25046515527486, 8.115672386637337,
    52.57202254002265, 7.073620667430696
]
    

hourly_dataframe = pd.DataFrame()

if os.path.exists("DataFrames/Photovoltaik/Photovoltaik_2015_2022_17Std_18Data_&_Wetter.csv"):
    
    hourly_dataframe = pd.read_csv("DataFrames/Photovoltaik/Photovoltaik_2015_2022_17Std_18Data_&_Wetter.csv")
    hourly_dataframe = hourly_dataframe.drop(hourly_dataframe.columns[[0]], axis=1)
    
else:


    hourly_dataframe = pd.DataFrame()

    for i in range(0, 17):
        
        params = {
            "latitude": coor[i * 2],
            "longitude": coor[i * 2 + 1],
            "start_date": "2015-01-01",
            "end_date": "2022-12-31",
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain",
                    "snowfall", "surface_pressure", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
                    "wind_direction_10m", "wind_direction_100m", "sunshine_duration", "shortwave_radiation",
                    "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "terrestrial_radiation"],
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
        hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
        hourly_rain = hourly.Variables(4).ValuesAsNumpy()
        hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(6).ValuesAsNumpy()
        hourly_vapour_pressure_deficit = hourly.Variables(7).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
        hourly_wind_speed_100m = hourly.Variables(9).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(10).ValuesAsNumpy()
        hourly_wind_direction_100m = hourly.Variables(11).ValuesAsNumpy()
        hourly_sunshine_duration = hourly.Variables(12).ValuesAsNumpy()
        hourly_shortwave_radiation = hourly.Variables(13).ValuesAsNumpy()
        hourly_direct_radiation = hourly.Variables(14).ValuesAsNumpy()
        hourly_diffuse_radiation = hourly.Variables(15).ValuesAsNumpy()
        hourly_direct_normal_irradiance = hourly.Variables(16).ValuesAsNumpy()
        hourly_terrestrial_radiation = hourly.Variables(17).ValuesAsNumpy()

        hourly_data = {}
        hourly_data["temperature_2m_" + str(i)] = hourly_temperature_2m
        hourly_data["relative_humidity_2m_" + str(i)] = hourly_relative_humidity_2m
        hourly_data["dew_point_2m_" + str(i)] = hourly_dew_point_2m
        hourly_data["apparent_temperature_" + str(i)] = hourly_apparent_temperature
        hourly_data["rain_" + str(i)] = hourly_rain
        hourly_data["snowfall_" + str(i)] = hourly_snowfall
        hourly_data["surface_pressure_" + str(i)] = hourly_surface_pressure
        hourly_data["vapour_pressure_deficit_" + str(i)] = hourly_vapour_pressure_deficit
        hourly_data["wind_speed_10m_" + str(i)] = hourly_wind_speed_10m
        hourly_data["wind_speed_100m_" + str(i)] = hourly_wind_speed_100m
        hourly_data["wind_direction_10m_" + str(i)] = hourly_wind_direction_10m
        hourly_data["wind_direction_100m_" + str(i)] = hourly_wind_direction_100m
        hourly_data["sunshine_duration_" + str(i)] = hourly_sunshine_duration
        hourly_data["shortwave_radiation_" + str(i)] = hourly_shortwave_radiation
        hourly_data["direct_radiation_" + str(i)] = hourly_direct_radiation
        hourly_data["diffuse_radiation_" + str(i)] = hourly_diffuse_radiation
        hourly_data["direct_normal_irradiance_" + str(i)] = hourly_direct_normal_irradiance
        hourly_data["terrestrial_radiation_" + str(i)] = hourly_terrestrial_radiation

        add_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe = pd.concat([hourly_dataframe, add_dataframe], axis=1)
        print(hourly_dataframe)
        
        
    print(hourly_dataframe)
    hourly_dataframe.to_csv("DataFrames/Photovoltaik/Photovoltaik_2015_2022_17Std_18Data.csv")

    # Add Colum from Photovoltaik Power
    hourly_dataframe['Photovoltaik'] = pd.read_csv("Power_Data/Photovoltaik_2015_2022.csv")
    hourly_dataframe.to_csv("DataFrames/Photovoltaik/Photovoltaik_2015_2022_17Std_18Data_&_Wetter.csv")



# Split Data into useable Datasets
x = hourly_dataframe.drop(columns = "Photovoltaik")
y = hourly_dataframe["Photovoltaik"]

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
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.add(InputLayer((306,)))
  model.add(Dense(units=hp_layer_1, activation='relu'))
  model.add(Dense(units=hp_layer_2, activation='relu'))
  model.add(Dense(units=hp_layer_3, activation='relu'))
  model.add(Dense(1, 'linear'))

  model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                loss='mse',
                metrics=['mse'])
  
  return model

model_name = 'autoF_2015_2022_17Stand_3Ly_18Data'

tuner = kt.Hyperband(model_builder,
                     objective="val_mse",
                     max_epochs=10,
                     factor=3,
                     directory='hyper_var_fitting/Photovoltaik',
                     project_name=model_name)


stop_early = EarlyStopping(monitor='val_loss', patience=3)
tuner.search(x_train_preprocessed, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = Model

if os.path.exists("models/Photovoltaik/" + model_name):
    model = load_model("models/Photovoltaik/" + model_name)
    
else:
    cp = ModelCheckpoint("models/Photovoltaik/" + model_name, save_best_only=True)
    model = tuner.hypermodel.build(best_hps)
    model.fit(x=x_train_preprocessed, y=y_train, validation_data=(x_test_preprocessed, y_test), callbacks=[cp], epochs=100)


y_pred_test = model.predict(x_test_preprocessed)


plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Photovoltaik Power")
plt.ylabel("Predicted Photovoltaik Power")
plt.show()


result = pd.DataFrame(y_test)
result['Photovoltaik'] = y_pred_test

mse = mean_squared_error(y_test, y_pred_test, squared=False)
print("MSE: ", mse)

plt.plot(result)
plt.legend(['Photovoltaik', 'Pred_Photovoltaik'])
plt.title("Mean_squared_error: " + mse.astype('str'))
plt.show()

# autoF_2015_2022_17Stand_3Ly_18Data
# MSE:  2935.985574267818

timestamp_range = pd.date_range(start="2018-01-01", 
                                end="2022-12-31 23:45", freq="H")


plt.plot(timestamp_range[:200], y_test[:200])
plt.plot(timestamp_range[:200], result[:200])
plt.legend(['Validation', 'Val Predictions'])
plt.show()


dataframe_2023 = pd.DataFrame()

if os.path.exists("DataFrames/Photovoltaik/Photovoltaik_2023_bis_Nov.csv"):
    
    dataframe_2023 = pd.read_csv("DataFrames/Photovoltaik/Photovoltaik_2023_bis_Nov.csv")
    dataframe_2023 = dataframe_2023.drop(dataframe_2023.columns[[0]], axis=1)
    
else:
    
    for i in range(0, 17):
            
        params = {
            "latitude": coor[i * 2],
            "longitude": coor[i * 2 + 1],
            "start_date": "2023-01-01",
            "end_date": "2023-11-30",
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain",
                    "snowfall", "surface_pressure", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
                    "wind_direction_10m", "wind_direction_100m", "sunshine_duration", "shortwave_radiation",
                    "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "terrestrial_radiation"],
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
        hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
        hourly_rain = hourly.Variables(4).ValuesAsNumpy()
        hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(6).ValuesAsNumpy()
        hourly_vapour_pressure_deficit = hourly.Variables(7).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
        hourly_wind_speed_100m = hourly.Variables(9).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(10).ValuesAsNumpy()
        hourly_wind_direction_100m = hourly.Variables(11).ValuesAsNumpy()
        hourly_sunshine_duration = hourly.Variables(12).ValuesAsNumpy()
        hourly_shortwave_radiation = hourly.Variables(13).ValuesAsNumpy()
        hourly_direct_radiation = hourly.Variables(14).ValuesAsNumpy()
        hourly_diffuse_radiation = hourly.Variables(15).ValuesAsNumpy()
        hourly_direct_normal_irradiance = hourly.Variables(16).ValuesAsNumpy()
        hourly_terrestrial_radiation = hourly.Variables(17).ValuesAsNumpy()

        hourly_data = {}
        hourly_data["temperature_2m_" + str(i)] = hourly_temperature_2m
        hourly_data["relative_humidity_2m_" + str(i)] = hourly_relative_humidity_2m
        hourly_data["dew_point_2m_" + str(i)] = hourly_dew_point_2m
        hourly_data["apparent_temperature_" + str(i)] = hourly_apparent_temperature
        hourly_data["rain_" + str(i)] = hourly_rain
        hourly_data["snowfall_" + str(i)] = hourly_snowfall
        hourly_data["surface_pressure_" + str(i)] = hourly_surface_pressure
        hourly_data["vapour_pressure_deficit_" + str(i)] = hourly_vapour_pressure_deficit
        hourly_data["wind_speed_10m_" + str(i)] = hourly_wind_speed_10m
        hourly_data["wind_speed_100m_" + str(i)] = hourly_wind_speed_100m
        hourly_data["wind_direction_10m_" + str(i)] = hourly_wind_direction_10m
        hourly_data["wind_direction_100m_" + str(i)] = hourly_wind_direction_100m
        hourly_data["sunshine_duration_" + str(i)] = hourly_sunshine_duration
        hourly_data["shortwave_radiation_" + str(i)] = hourly_shortwave_radiation
        hourly_data["direct_radiation_" + str(i)] = hourly_direct_radiation
        hourly_data["diffuse_radiation_" + str(i)] = hourly_diffuse_radiation
        hourly_data["direct_normal_irradiance_" + str(i)] = hourly_direct_normal_irradiance
        hourly_data["terrestrial_radiation_" + str(i)] = hourly_terrestrial_radiation

        add_dataframe = pd.DataFrame(data = hourly_data)
        dataframe_2023 = pd.concat([dataframe_2023, add_dataframe], axis=1)
        #time.sleep(30)
        
    dataframe_2023.to_csv("DataFrames/Photovoltaik/Photovoltaik_2023_bis_Nov.csv")
        
print(dataframe_2023)
dataframe_2023_preprocessed = preproccesor(dataframe_2023)

pred_2023 = model.predict(dataframe_2023_preprocessed)
off_shore_2023 = pd.read_csv("Power_Data/Photovoltaik_2023_bis_Nov.csv")


timestamp_range_2023 = pd.date_range(start="2023-01-01", 
                                end="2023-11-30 23:00", freq="H")

print(timestamp_range_2023)
timestamp_range_as_np_2023 = timestamp_range_2023.to_numpy()

plt.plot(timestamp_range_as_np_2023[:500], off_shore_2023[:500])
plt.plot(timestamp_range_as_np_2023[:500], pred_2023[:500])
plt.legend(['Photovoltaik_2023', 'pred_2023'])
plt.show()