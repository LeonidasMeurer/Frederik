'''
from deutschland import smard
from deutschland.smard.api import default_api
from deutschland.smard.model.indices import Indices
from pprint import pprint


import pandas as pd
import numpy as np



# Defining the host is optional and defaults to https://www.smard.de/app
# See configuration.py for a list of all supported configuration parameters.
configuration = smard.Configuration(
    host = "https://www.smard.de/app"
)


# Enter a context with an instance of the API client
with smard.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)
    filter = 1223 # int | Mögliche Filter:   * `1223` - Stromerzeugung: Braunkohle   * `1224` - Stromerzeugung: Kernenergie   * `1225` - Stromerzeugung: Wind Offshore   * `1226` - Stromerzeugung: Wasserkraft   * `1227` - Stromerzeugung: Sonstige Konventionelle   * `1228` - Stromerzeugung: Sonstige Erneuerbare   * `4066` - Stromerzeugung: Biomasse   * `4067` - Stromerzeugung: Wind Onshore   * `4068` - Stromerzeugung: Photovoltaik   * `4069` - Stromerzeugung: Steinkohle   * `4070` - Stromerzeugung: Pumpspeicher   * `4071` - Stromerzeugung: Erdgas   * `410` - Stromverbrauch: Gesamt (Netzlast)   * `4359` - Stromverbrauch: Residuallast   * `4387` - Stromverbrauch: Pumpspeicher   * `4169` - Marktpreis: Deutschland/Luxemburg   * `5078` - Marktpreis: Anrainer DE/LU   * `4996` - Marktpreis: Belgien   * `4997` - Marktpreis: Norwegen 2   * `4170` - Marktpreis: Österreich   * `252` - Marktpreis: Dänemark 1   * `253` - Marktpreis: Dänemark 2   * `254` - Marktpreis: Frankreich   * `255` - Marktpreis: Italien (Nord)   * `256` - Marktpreis: Niederlande   * `257` - Marktpreis: Polen   * `258` - Marktpreis: Polen   * `259` - Marktpreis: Schweiz   * `260` - Marktpreis: Slowenien   * `261` - Marktpreis: Tschechien   * `262` - Marktpreis: Ungarn   * `3791` - Prognostizierte Erzeugung: Offshore   * `123` - Prognostizierte Erzeugung: Onshore   * `125` - Prognostizierte Erzeugung: Photovoltaik   * `715` - Prognostizierte Erzeugung: Sonstige   * `5097` - Prognostizierte Erzeugung: Wind und Photovoltaik   * `122` - Prognostizierte Erzeugung: Gesamt 
    filter_copy = 1223 # int | Muss dem Wert von \"filter\" entsprechen. (Kaputtes API-Design) 
    region_copy = "DE" # str | Muss dem Wert von \"region\" entsprechen. (Kaputtes API-Design) 
    timestamp = 1420412400000 # int |
    region='DE'
    resolution='quarterhour'
    
    # example passing only required values which don't have defaults set
    try:
        # Indizes
        api_response = api_instance.filter_region_index_resolution_json_get(filter, region, resolution)
        pprint(api_response)
    except smard.ApiException as e:
        print("Exception when calling DefaultApi->chart_data_filter_region_index_resolution_json_get: %s\n" % e)
        
        
    try:
        # Zeitreihendaten
        api_response = api_instance.filter_region_filter_copy_region_copy_resolution_timestamp_json_get(
            filter, filter_copy, region_copy, timestamp, region, resolution)
        pprint(api_response)
    except smard.ApiException as e:
        print("Exception when calling DefaultApi->table_data_filter_region_filter_copy_region_copy_quarterhour_timestamp_json_get: %s\n" % e)'''