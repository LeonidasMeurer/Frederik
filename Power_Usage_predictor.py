from datetime import timedelta

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from keras.models import load_model

from SMARD_api import DataFrame_Smard_Api_Gen




class PowerPredictor:
    
  def get_Prediction(self):
    # GET DATA
    smard_api = DataFrame_Smard_Api_Gen()
    hourly_dataframe = smard_api.get_smard_data()

    ###TIME FRAME MUSS STIMMEN SONST PRED SCHLECHT!!!

    hourly_dataframe['Seconds'] = hourly_dataframe.index.map(pd.Timestamp.timestamp)

    day = 60*60*24
    year = 365.2425*day

    hourly_dataframe['Day sin'] = np.sin(hourly_dataframe['Seconds'] * (2* np.pi / day))
    hourly_dataframe['Day cos'] = np.cos(hourly_dataframe['Seconds'] * (2 * np.pi / day))
    hourly_dataframe['Year sin'] = np.sin(hourly_dataframe['Seconds'] * (2 * np.pi / year))
    hourly_dataframe['Year cos'] = np.cos(hourly_dataframe['Seconds'] * (2 * np.pi / year))

    hourly_dataframe = hourly_dataframe.drop('Seconds', axis=1)
    hourly_dataframe.reset_index()


    # STANDARDIZE
    scaler = StandardScaler().fit(hourly_dataframe)
    hourly_dataframe_standard = scaler.transform(hourly_dataframe)




    # BUILD 3D np_array
    x = []

    df_as_np = hourly_dataframe_standard
    row = [a for a in df_as_np]
    x.append(row)
  
    x = np.array(x)


    model = load_model('models/Power/Power_model')

    normal_pred = model.predict(x)

    normal_pred_df=pd.DataFrame(np.hstack(normal_pred), columns=hourly_dataframe.columns)

    # BUILD RESULT DATAFRAME
    start_hour = hourly_dataframe.index[-1] + timedelta(hours=1)

    timestamp_range_pred = pd.date_range(start=start_hour, freq="H", periods=168) 
    normal_pred_df["index"] = timestamp_range_pred
    normal_pred_df = normal_pred_df.set_index("index")


    normal_pred_df = normal_pred_df.drop('Day sin', axis=1)
    normal_pred_df = normal_pred_df.drop('Day cos', axis=1)
    normal_pred_df = normal_pred_df.drop('Year sin', axis=1)
    normal_pred_df = normal_pred_df.drop('Year cos', axis=1)
    normal_pred_df = normal_pred_df.drop('Wind Offshore', axis=1)
    normal_pred_df = normal_pred_df.drop('Wind Onshore', axis=1)
    normal_pred_df = normal_pred_df.drop('Photovoltaik', axis=1)

    '''normal_pred_df.plot(subplots=True, figsize=(6, 6))
    plt.show()'''
    
    return normal_pred_df

'''pp = PowerPredictor()
plt.plot(pp.get_Prediction())
plt.show()'''