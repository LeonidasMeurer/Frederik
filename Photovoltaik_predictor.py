import openmeteo_requests

import requests_cache
import pandas as pd

from retry_requests import retry

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import load_model

default_coor = [
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

class PhotovoltaikPredictor:
    
    def __init__(self):
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = retry_session)

        
    def get_Prediction(self, forecast_days, latitude=0, longitude=0, pWk=1):
        """Returns MWh prediction for Germany on default can use lat and long for speficig forecast.
        You can also give a pWk Number depending on the PV-System, it defaults to 1.
        For Home Users multiply the result with 1000 to get KWh!!

        Args:
            forecast_days (int): forecast_days
            latitude (float, optional): Use following format: 52.57202254002265. Defaults to 0.
            longitude (float, optional): Use following format: 7.073620667430696. Defaults to 0.
            pWk (int, optional): _description_. Defaults to 1.

        Raises:
            Exception: latitude missing if longitude is given
            Exception: longitude missing if latitude is given

        Returns:
            Dataframe
        """

        
        if latitude==0 and longitude!=0:
            raise Exception("Please, select longitude")
        elif latitude!=0 and longitude==0:
            raise Exception("Please, select latitude")

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://api.open-meteo.com/v1/forecast"


        coor=[]

        if (latitude != 0 and longitude != 0):
            for i in range(0, 17):
                coor.append(latitude)
                coor.append(longitude)
        else:
            coor = default_coor
        

        hourly_dataframe = pd.DataFrame()
        

        for i in range(0, int(len(coor)/2)):
                
            params = {
                "latitude": coor[i * 2],
                "longitude": coor[i * 2 + 1],
                "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain",
                        "snowfall", "surface_pressure", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
                        "wind_direction_10m", "wind_direction_100m", "sunshine_duration", "shortwave_radiation",
                        "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "terrestrial_radiation"],
                "wind_speed_unit": "ms",
                "forecast_days": forecast_days
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
            

        # STANDARDIZE
        scaler = StandardScaler().fit(hourly_dataframe)
        hourly_dataframe_preprocessed = scaler.transform(hourly_dataframe)

        model = load_model('models/Photovoltaik/Photovoltaik_model')

        prediction = model.predict(hourly_dataframe_preprocessed)
        
        
        
        # CALCULATE FOR SINGLE HOUSE HOLD
        insgesamt = 0
         
        np_array = []
        for value in prediction:
                np_array.append(value)
        for value in prediction:
                insgesamt =+ np_array.pop()

        
        '''
        1. Insgesamt Strom pred
        2. Erwarted on average: 2.8 kWH * tage pro pWk
        
        Erwarted / Insgesamt * 1000 (Mwh zu Kwh) = Faktor
        
        Faktor * jedes_pred_value = Vorhersage pro pWk
        
        Mittles dieses dreisatzes lässt sich das model auch anwenden um den durchschnitlichen zu erwartbaren kwH Wert für ein Haus pro pWk zu berechnen
        '''
        
        
        factor = ((2.8 * forecast_days) / (insgesamt * 1000))

        
        
        refactored_for_single_House = []
        
        if longitude!=0 and latitude!=0:
            for value in prediction:
                refactored_for_single_House.append((value * factor) * pWk)

        

        timestamp_range = pd.date_range(start=pd.to_datetime(hourly.Time(), unit = "s"), 
                                        end=pd.to_datetime(hourly.TimeEnd(), unit = "s"),
                                        freq=pd.Timedelta(seconds = hourly.Interval()),
                                        inclusive='left')

               
        result = pd.DataFrame()
        
        '''result = pd.DataFrame(data=prediction, index=timestamp_range_fitted)
        plt.plot(timestamp_range_fitted, result)
        plt.show()
        
        result = pd.DataFrame(data=refactored_for_single_House, index=timestamp_range_fitted)
        plt.plot(timestamp_range_fitted, result)
        plt.show()'''
        
       
        
        if latitude==0:
            result = pd.DataFrame(data=prediction, index=timestamp_range, columns=['Photovoltaik'])
        else:
            result = pd.DataFrame(data=refactored_for_single_House, index=timestamp_range, columns=['Photovoltaik'])
            for a in refactored_for_single_House:
                insgesamt = refactored_for_single_House.pop()
            
          
        return result
    
    
'''pp_1 = PhotovoltaikPredictor()
pp_1.get_Prediction(forecast_days=10, latitude=53.25046515527486, longitude=8.115672386637337, pWk=1)
pp_1.get_Prediction(forecast_days=1)'''

