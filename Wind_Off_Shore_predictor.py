import openmeteo_requests

import requests_cache
import pandas as pd

from retry_requests import retry

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

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
                "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain", "snowfall",
                                    "pressure_msl", "surface_pressure", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
                                    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m"],
                "wind_speed_unit": "ms",
                "forecast_days": forecast_days,
                "timezone": "Europe/Berlin"
            }
            responses = self.openmeteo.weather_api(url, params=params)

            # Process first location. Add a for-loop for multiple locations or weather models
            response = responses[0]

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




        # STANDARDIZE
        scaler = StandardScaler().fit(hourly_dataframe)
        hourly_dataframe_preprocessed = scaler.transform(hourly_dataframe)
        

        model = load_model('models/Off_Shore/Off_Shore_model')

        prediction = model.predict(hourly_dataframe_preprocessed)

        timestamp_range = pd.date_range(start=pd.to_datetime(hourly.Time(), unit = "s"), 
                                        end=pd.to_datetime(hourly.TimeEnd(), unit = "s"),
                                        freq=pd.Timedelta(seconds = hourly.Interval()),
                                        inclusive = "left")


        result = pd.DataFrame(data=prediction, index=timestamp_range, columns=['Wind Offshore'])
        '''plt.plot(result)
        plt.show()'''
        
        
        '''result.to_csv('csv_gen/off_shore_7days')'''
          
        return result
    
    
'''pp_1 = WindOffShorePredictor()
pp_1.get_Prediction(7)'''