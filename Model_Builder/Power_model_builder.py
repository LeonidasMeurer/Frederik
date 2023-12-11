import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError, MeanSquaredError

from keras.models import Sequential, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping





# GET DATA
hourly_dataframe = pd.read_csv("Power_Data/Power_2015_2022.csv")
timestamp_range = pd.date_range(start="2015-01-01", end="2022-12-31 23:45", freq="H")

hourly_dataframe["index"] = timestamp_range
hourly_dataframe = hourly_dataframe.set_index("index")
hourly_dataframe['Seconds'] = hourly_dataframe.index.map(pd.Timestamp.timestamp)

day = 60*60*24
year = 365.2425*day

hourly_dataframe['Day sin'] = np.sin(hourly_dataframe['Seconds'] * (2* np.pi / day))
hourly_dataframe['Day cos'] = np.cos(hourly_dataframe['Seconds'] * (2 * np.pi / day))
hourly_dataframe['Year sin'] = np.sin(hourly_dataframe['Seconds'] * (2 * np.pi / year))
hourly_dataframe['Year cos'] = np.cos(hourly_dataframe['Seconds'] * (2 * np.pi / year))

hourly_dataframe = hourly_dataframe.drop('Seconds', axis=1)
hourly_dataframe.reset_index()

n = len(hourly_dataframe)
train_df = hourly_dataframe[0:int(n*0.7)]
val_df = hourly_dataframe[int(n*0.7):int(n*0.9)]
test_df = hourly_dataframe[int(n*0.9):]

num_features = hourly_dataframe.shape[1]



# SPLIT DATA
def df_to_X_y(df, input_size, output_size):
  df_as_np = df.to_numpy()
  x = []
  y = []
  for i in range(len(df_as_np)-(input_size + output_size)):
    row = [a for a in df_as_np[i:i+input_size]]
    x.append(row)

    label = [a for a in df_as_np[(input_size + i) : (i + input_size +  output_size)]]
    y.append(label)
  
  return np.array(x), np.array(y)

INPUTE_SIZE = 168 * 4
OUTPUT_SIZE = 168
x, y = df_to_X_y(hourly_dataframe, INPUTE_SIZE, OUTPUT_SIZE)



dates_train, x_train, y_train = hourly_dataframe.index[0:int(n*0.7)], x[0:int(n*0.7)], y[0:int(n*0.7)]
dates_val, x_val, y_val = hourly_dataframe.index[0:int(n*0.7)], x[int(n*0.7):int(n*0.9)], y[int(n*0.7):int(n*0.9)]
dates_test, x_test, y_test = hourly_dataframe.index[0:int(n*0.7)], x[int(n*0.9):], y[int(n*0.9):]


# STANDARDIZE DATA

train_scalers = {}
for i in range(x_train.shape[1]):
    train_scalers[i] = StandardScaler().fit(x_train[:, i, :])
    x_train[:, i, :] = train_scalers[i].transform(x_train[:, i, :]) 


val_scalers = {}
for i in range(x_val.shape[1]):
    val_scalers[i] = StandardScaler().fit(x_val[:, i, :])
    x_val[:, i, :] = val_scalers[i].transform(x_val[:, i, :]) 
    
test_scalers = {} 
for i in range(x_test.shape[1]):
    test_scalers[i] = StandardScaler().fit(x_test[:, i, :])
    x_test[:, i, :] = test_scalers[i].transform(x_test[:, i, :])




# MAKE MODEL

model = Sequential([tf.keras.layers.Input((INPUTE_SIZE, num_features)),
                    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dense(OUTPUT_SIZE * num_features, kernel_initializer=tf.initializers.zeros()),
                    tf.keras.layers.Reshape([OUTPUT_SIZE, num_features])
                    ])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
cp = ModelCheckpoint('models/Power/Power_model', save_best_only=True)


model.compile(loss=MeanSquaredError() ,optimizer=Adam(), metrics=[MeanAbsoluteError()])
model.fit(x_train, y_train ,validation_data=(x_val, y_val), epochs=100, callbacks=[early_stopping, cp])


model = load_model('models/Power/Power_model')



# TEST
START_WEEK = 30

normal_pred = model.predict(x_test[START_WEEK : START_WEEK + 1])
y_act = y_test[START_WEEK : START_WEEK + 1]


y_act_df=pd.DataFrame(np.hstack(y_act), columns=hourly_dataframe.columns)
y_act_df = y_act_df.drop('Day sin', axis=1)
y_act_df = y_act_df.drop('Day cos', axis=1)
y_act_df = y_act_df.drop('Year sin', axis=1)
y_act_df = y_act_df.drop('Year cos', axis=1)



normal_pred_df=pd.DataFrame(np.hstack(normal_pred), columns=hourly_dataframe.columns)
normal_pred_df = normal_pred_df.drop('Day sin', axis=1)
normal_pred_df = normal_pred_df.drop('Day cos', axis=1)
normal_pred_df = normal_pred_df.drop('Year sin', axis=1)
normal_pred_df = normal_pred_df.drop('Year cos', axis=1)

print(normal_pred_df)


offshore = pd.DataFrame(y_act_df['Wind Offshore'])
offshore['Prediction'] = normal_pred_df['Wind Offshore']

onshore = pd.DataFrame(y_act_df['Wind Onshore'])
onshore['Prediction'] = normal_pred_df['Wind Onshore']

photovoltaik = pd.DataFrame(y_act_df['Photovoltaik'])
photovoltaik['Prediction'] = normal_pred_df['Photovoltaik']

preis = pd.DataFrame(y_act_df['Residuallast'], columns=['Residuallast'])
preis['Prediction'] = normal_pred_df['Residuallast']

power_usage = pd.DataFrame(y_act_df['Power Usage'])
power_usage['Prediction'] = normal_pred_df['Power Usage']


#define subplot layout
fig, axes = plt.subplots(nrows=5, ncols=1)

#add DataFrames to subplots
offshore.plot(ax=axes[0])
onshore.plot(ax=axes[1])
photovoltaik.plot(ax=axes[2])
preis.plot(ax=axes[3])
power_usage.plot(ax=axes[4])
plt.show()
#define subplot layout

#add DataFrames to subplots
normal_pred_df.plot(subplots=True, figsize=(6, 6))
plt.show()

