from deutschland import smard
from deutschland.smard.api import default_api
from datetime import datetime

import pandas as pd


filter_array = [410, 4359, 4387, 1225, 4067, 4068]
filter_labels = ['Power Usage','Residuallast','Pumpspeicher','Wind Offshore','Wind Onshore','Photovoltaik']



df_result = pd.DataFrame(columns=filter_labels)

class DataFrame_Smard_Api_Gen:
    
    def __init__(self):
        
        # Defining the host is optional and defaults to https://www.smard.de/app
        # See configuration.py for a list of all supported configuration parameters.
        self.configuration = smard.Configuration(
            host = "https://www.smard.de/app"
        )


        # Enter a context with an instance of the API client
        with smard.ApiClient() as api_client:
            # Create an instance of the API class
            self.api_instance = default_api.DefaultApi(api_client)
            self.filter = 1225 # int | Mögliche Filter:   * `1223` - Stromerzeugung: Braunkohle   * `1224` - Stromerzeugung: Kernenergie   * `1225` - Stromerzeugung: Wind Offshore   * `1226` - Stromerzeugung: Wasserkraft   * `1227` - Stromerzeugung: Sonstige Konventionelle   * `1228` - Stromerzeugung: Sonstige Erneuerbare   * `4066` - Stromerzeugung: Biomasse   * `4067` - Stromerzeugung: Wind Onshore   * `4068` - Stromerzeugung: Photovoltaik   * `4069` - Stromerzeugung: Steinkohle   * `4070` - Stromerzeugung: Pumpspeicher   * `4071` - Stromerzeugung: Erdgas   * `410` - Stromverbrauch: Gesamt (Netzlast)   * `4359` - Stromverbrauch: Residuallast   * `4387` - Stromverbrauch: Pumpspeicher   * `4169` - Marktpreis: Deutschland/Luxemburg   * `5078` - Marktpreis: Anrainer DE/LU   * `4996` - Marktpreis: Belgien   * `4997` - Marktpreis: Norwegen 2   * `4170` - Marktpreis: Österreich   * `252` - Marktpreis: Dänemark 1   * `253` - Marktpreis: Dänemark 2   * `254` - Marktpreis: Frankreich   * `255` - Marktpreis: Italien (Nord)   * `256` - Marktpreis: Niederlande   * `257` - Marktpreis: Polen   * `258` - Marktpreis: Polen   * `259` - Marktpreis: Schweiz   * `260` - Marktpreis: Slowenien   * `261` - Marktpreis: Tschechien   * `262` - Marktpreis: Ungarn   * `3791` - Prognostizierte Erzeugung: Offshore   * `123` - Prognostizierte Erzeugung: Onshore   * `125` - Prognostizierte Erzeugung: Photovoltaik   * `715` - Prognostizierte Erzeugung: Sonstige   * `5097` - Prognostizierte Erzeugung: Wind und Photovoltaik   * `122` - Prognostizierte Erzeugung: Gesamt 
            self.filter_copy = 1225 # int | Muss dem Wert von \"filter\" entsprechen. (Kaputtes API-Design) 
            self.region_copy = "DE" # str | Muss dem Wert von \"region\" entsprechen. (Kaputtes API-Design) 
            self.region='DE'
            self.resolution='hour'
            
            
            
    
    def get_smard_data(self):
        
            df_result = pd.DataFrame(columns=filter_labels)
              
            try:
                # Indizes
                api_response = self.api_instance.filter_region_index_resolution_json_get(self.filter, self.region, self.resolution)
            except smard.ApiException as e:
                print("Exception when calling DefaultApi->chart_data_filter_region_index_resolution_json_get: %s\n" % e)
                
                
            timestamp_api_response = []
            
            # Check for Monday
            date = datetime.today()
            if date.today().weekday() == 0:
                timestamp_api_response = api_response.timestamps[-6:-1]
            else:
                timestamp_api_response = api_response.timestamps[-5:]
            
            try:
                # Zeitreihendaten
                api_response = self.api_instance.filter_region_filter_copy_region_copy_resolution_timestamp_json_get(
                    self.filter, self.filter_copy, self.region_copy, timestamp_api_response[-1], self.region, self.resolution)
            except smard.ApiException as e:
                print("Exception when calling DefaultApi->table_data_filter_region_filter_copy_region_copy_quarterhour_timestamp_json_get: %s\n" % e)
                
            api_array = api_response.series
            last_available_date = api_array[-1]
            

            
            counter = 0
            for filter_value in filter_array:
            
                build_array = []
            
                for timestamp in timestamp_api_response:

                    try:
                        # Zeitreihendaten
                        api_response = self.api_instance.filter_region_filter_copy_region_copy_resolution_timestamp_json_get(
                            filter_value, filter_value, self.region_copy, timestamp, self.region, self.resolution)
                    except smard.ApiException as e:
                        print("Exception when calling DefaultApi->table_data_filter_region_filter_copy_region_copy_quarterhour_timestamp_json_get: %s\n" % e)
                    api_array = api_response.series
                    for value in api_array:
                        build_array.append(value[1])
                                            
                  
                
                df_result[filter_labels[counter]] = build_array      
                counter = counter + 1
                
            
            timestamp_range = pd.date_range(end=(datetime.fromtimestamp(last_available_date[0] / 1000)),
                                            freq="H", periods=len(df_result))
            df_result["index"] = timestamp_range
            df_result = df_result.set_index("index")    
    
            return df_result[-672:]
        
        