import openmeteo_requests

import requests_cache
import pandas as pd
import numpy as np

from retry_requests import retry

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import load_model


class WindOffShorePredictor:
    
    def __init__(self):
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = retry_session)
        

    def get_Prediction(self, forecast_days):
    
        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://api.open-meteo.com/v1/forecast"

        coor = [
            55.1226, 6.48353,
            54.173701, 6.25818,
            54.57597, 13.06216
        ]

        hourly_dataframe = pd.DataFrame()

        for i in range(0, 3):
            
            params = {
                "latitude": coor[i * 2],
                "longitude": coor[i * 2 + 1],
                "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_speed_120m", "wind_direction_10m", "wind_direction_120m", "wind_gusts_10m"],
                "wind_speed_unit": "ms",
                "forecast_days": forecast_days
            }
            responses = self.openmeteo.weather_api(url, params=params)

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



        # Normalize Data
        scaler = StandardScaler().fit(hourly_dataframe)

        def preproccesor(X):
            A = np.copy(X)
            A = scaler.transform(A)
            return A

        hourly_dataframe_preprocessed = preproccesor(hourly_dataframe)

        model = load_model('models/Off_Shore/autoF_2018_2022_3Stand')

        prediction = model.predict(hourly_dataframe_preprocessed)

        timestamp_range = pd.date_range(start=pd.to_datetime(hourly.Time(), unit = "s"), 
                                        end=pd.to_datetime(hourly.TimeEnd(), unit = "s"), freq=pd.Timedelta(seconds = hourly.Interval()))
        timestamp_range_fitted = timestamp_range.delete([-1])

        result = pd.DataFrame(data=prediction, index=timestamp_range_fitted)
        result.to_json()
        '''plt.plot(result)
        plt.show()'''
        
        print(result)
          
        return result.to_json(orient="columns")
    
    
'''pp_1 = WindOffShorePredictor()
pp_1.get_Prediction(3)'''