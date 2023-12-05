import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', None)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from keras.callbacks import ModelCheckpoint


from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense
from copy import deepcopy
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError

data = pd.read_csv("Power_Data/HomeC.csv",low_memory=False)
data = data[:-1] #delete last row  (NaNs)
data.info()

#Convert Unix timestamp to datetime, use sample frequency of minutes and make it dataframe index
data['time'] = pd.to_datetime(data['time'], unit='s')
data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min'))
data = data.set_index('time')

#Delete '[kW]' in columns name, sum similar consumtions and delete 'summary' column
data.columns = [i.replace(' [kW]', '') for i in data.columns]
data['Furnace'] = data[['Furnace 1','Furnace 2']].sum(axis=1)
data['Kitchen'] = data[['Kitchen 12','Kitchen 14','Kitchen 38']].sum(axis=1) #We could also use the mean 
data.drop(['Furnace 1','Furnace 2','Kitchen 12','Kitchen 14','Kitchen 38','icon','summary'], axis=1, inplace=True)

#Replace invalid values in column 'cloudCover' with backfill method
data['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)
data['cloudCover'] = data['cloudCover'].astype('float')

#Reorder columns
data = data[['use', 'gen', 'House overall', 'Dishwasher', 'Home office', 'Fridge', 'Wine cellar', 'Garage door', 'Barn',
             'Well', 'Microwave', 'Living room', 'Furnace', 'Kitchen', 'Solar', 'temperature', 'humidity', 'visibility', 
             'apparentTemperature', 'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity', 
             'dewPoint', 'precipProbability']]
data.head(2)

#Reduce size of dataframe with only the columns we are interested in
data_daily = data[['House overall', 'Furnace', 'Living room', 'Barn', 'temperature', 'humidity',
                   'apparentTemperature', 'pressure', 'cloudCover','windBearing', 'precipIntensity',
                   'dewPoint', 'precipProbability']]
#Rescale
data_daily = data_daily.resample('D').mean()
#Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
data_daily[data_daily.columns[1:]] = scaler.fit_transform(data_daily[data_daily.columns[1:]])
scaler_target = MinMaxScaler(feature_range=(0, 1))
data_daily[['House overall']] = scaler_target.fit_transform(data_daily[['House overall']])

size = int(len(data_daily)*0.7)
data_daily_train = data_daily[:size]
data_daily_test = data_daily[size:]
X_train, X_test = [], []
Y_train, Y_test = [], []
n_past=1
n_future=1
for i in range(n_past, len(data_daily_train)-n_future+1):
    X_train.append(data_daily_train.iloc[i-n_past:i, 0:data_daily.shape[1]])
    Y_train.append(data_daily_train.iloc[i+n_future-1:i+n_future, 0])
for i in range(n_past, len(data_daily_test)-n_future+1):
    X_test.append(data_daily_test.iloc[i-n_past:i, 0:data_daily_test.shape[1]])
    Y_test.append(data_daily_test.iloc[i+n_future-1:i+n_future, 0])
    
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = np.array(X_test), np.array(Y_test)

print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
print('Y_train shape', Y_train.shape)
print('Y_test shape', Y_test.shape)

model = Sequential()
model.add(LSTM(25, activation='relu', return_sequences = False, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(LSTM(50, activation='relu', return_sequences = True))
#model.add(LSTM(15, activation='relu', return_sequences = False))
#model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()

cp = ModelCheckpoint(filepath='models/LSTM/Smart_House', save_best_only=True)
# model_fit = model.fit(X_train, Y_train,  epochs=60 )

model = load_model('models/LSTM/Smart_House')
# loss: 0.0149

Train_pred = model.predict(X_train, verbose=0)
Y_pred = model.predict(X_test, verbose=0)

'''plt.plot(model_fit.history['loss'], label='Train loss')
#plt.plot(model_fit.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
print('Train MSE minimum:', min(model_fit.history['loss']))
#print('Validation MSE minimum:', min(model_fit.history['val_loss']))'''

#Invert scaling
data_daily[['House overall']] = scaler_target.inverse_transform(data_daily[['House overall']])
Y_pred = scaler_target.inverse_transform(Y_pred)
Train_pred = scaler_target.inverse_transform(Train_pred)

plt.figure(figsize=(15,4))
plt.plot(data_daily[['House overall']][size:-1].values)
plt.plot(Y_pred)
plt.show()
np.sqrt(mean_squared_error(Y_pred[:,0].tolist(), data_daily[['House overall']][size:-1].values))

Y_pred_series = pd.Series(Y_pred.flatten().tolist(), index=data_daily['House overall'][size:-n_past].index)
Train_pred_series = pd.Series(Train_pred.flatten().tolist(), index=data_daily['House overall'][n_past:size].index)
plt.figure(figsize=(15,4))
plt.plot(data_daily['House overall'][:-n_past], c='blue', label='data')
plt.plot(Y_pred_series, c='red', label='model test')
plt.plot(Train_pred_series, c='green', label='model train')
plt.legend()
plt.grid(), plt.margins(x=0);
plt.show()
Y_test = data_daily['House overall'][size:-n_past]

# calcolo errore
print('MSE: %.5f' % (mean_squared_error(Y_pred, Y_test)))
print('RMSE: %.5f' % np.sqrt(mean_squared_error(Y_pred, Y_test)))
MAE = mean_absolute_error(Y_test, Y_pred)
MAPE = np.mean(np.abs(Y_pred[:,0] - Y_test.values)/np.abs(Y_test.values))
#MASE = np.mean(np.abs(Y_test - Y_pred))/(np.abs(np.diff(X_train)).sum()/(len(X_train)-1))
print('MAE: %.3f' % MAE)
print('MAPE: %.3f' %MAPE)
#print('MASE: %.3f' %MASE)
print('R^2 score: %.3f' % r2_score(Y_test, Y_pred))



recursive_predictions = []

last_window = deepcopy(X_train[-1])

timestamp_range = pd.date_range(start="2016-09-02", 
                                end="2022-12-31", freq="D")
print(timestamp_range)
print(Train_pred_series)
print(Y_pred_series)

for target_date in range(0, 5):
  next_prediction = model.predict(np.array([last_window])).flatten()
  recursive_predictions.append(next_prediction)
  last_window = np.roll(last_window, -1)
  last_window[-1] = next_prediction

recursive_predictions_series = pd.Series(np.array(recursive_predictions).flatten().tolist())
  
plt.figure(figsize=(15,4))
plt.plot(timestamp_range[:5], recursive_predictions_series)
plt.plot(Y_pred_series, c='red', label='model test')
plt.plot(Train_pred_series, c='green', label='model train')
plt.legend()
plt.grid(), plt.margins(x=0);
plt.show()  
