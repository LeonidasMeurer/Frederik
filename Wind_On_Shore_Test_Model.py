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



class WindOnShorePredictor:
    
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

        hourly_dataframe = pd.DataFrame()

        for i in range(0, 15):
            
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

        model = load_model('models/On_Shore/autoF_2015_2022_14Stand')

        prediction = model.predict(hourly_dataframe_preprocessed)


        timestamp_range = pd.date_range(start=pd.to_datetime(hourly.Time(), unit = "s"), 
                                        end=pd.to_datetime(hourly.TimeEnd(), unit = "s"), freq=pd.Timedelta(seconds = hourly.Interval()))
        timestamp_range_fitted = timestamp_range.delete([-1])

        result = pd.DataFrame(data=prediction, index=timestamp_range_fitted)
        '''plt.plot(result)
        plt.show()'''

        print(result)

        return result.to_json(orient="columns")
    
'''pp_1 = WindOnShorePredictor()
pp_1.get_Prediction(3)
'''