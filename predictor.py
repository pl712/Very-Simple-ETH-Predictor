import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import requests
import json
import matplotlib.pyplot as plt
from stockstats import StockDataFrame
assert format(tf.__version__) == '2.8.0'

#--------
# Gathering Data

pd.set_option('display.float_format', lambda x: '%.2f' % x)
data = requests.get('https://min-api.cryptocompare.com/data/histoday?fsym=ETH&tsym=USDT&limit=1500') #Data from 2018.2.17 to Today(2022.3.27)
df = pd.DataFrame(json.loads(data.content)['Data'])
df = df.rename(columns={'volumeto': 'volume', 'time': 'date'})

stock = StockDataFrame.retype(df)
stock.index = pd.to_datetime(stock.index, unit='s')

stock.get('close_12_ema'); stock.get('close_15_ema'); stock.get('boll');
stock["boll-close"] = stock["boll_ub"] - stock["close"]
stock["close-boll"] = stock["close"] - stock["boll_lb"]
stock = stock[["open", "volume", "high", "low", "close", "close_12_ema", "close_15_ema", "boll", "boll-close", "close-boll"]]
# These factors are used as they demonstrate at least medium to strong levels of correlation

stock = stock.tail(stock.shape[0] -1) #First row lack certain elements, so we drop it
stock.dropna()
stock.round(2) #Round to 2 decimals

#--------
# Debugging

#Correlation Heatmap (Debug.) -> Relatively strong relationships
# def heatmap():
  # import seaborn as sns
  # plt.figure(figsize=(16, 6))
  # sns.heatmap(stock.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG');

# Data Distribution (Debug.) -> use MinMax
# from scipy import stats
# def find_corr():
  # for i in stock.drop(columns=['close']).columns:
  #   x = stats.ks_2samp(stock['close'].values, stock[i].values)
  #   print(i, " is : ", x)  
  
#--------
# Normalization
List_of_scaler = {}

for i in stock.columns:
  scaler = MinMaxScaler()
  scaler.fit(stock[i].values.reshape(-1, 1))
  stock[i] = scaler.transform(stock[i].values.reshape(-1, 1))
  List_of_scaler[i] = scaler
  
#--------
# Slicing for LSTM
price = stock["close"]
features = stock.drop(columns = ["close"])

def make_data(dataX, dataY, batch_size):
  x, y = [], []
  for i in range(len(dataX)):
    pt = i + batch_size
    if pt > len(dataX)-1:
      break
    x.append(dataX[i : pt])
    y.append(dataY[pt])
  return np.array(x), np.array(y)

#--------
# Determining Train and Test Batches
TEST_RATIO = 0.2
BATCH = 64

X, Y = make_data(features, price, BATCH)
xTrain, xTest = train_test_split(X, test_size=TEST_RATIO, shuffle=False)
yTrain, yTest = train_test_split(Y, test_size=TEST_RATIO, shuffle=False)

n_info = xTrain.shape[0] # Number of combinations of said features, i guess | 'info' may be misleading
n_steps = xTrain.shape[1]
n_feat = xTrain.shape[2]

#--------
#Model and hyper params
LSTM_NEURONS = 125
DROPOUT = 0.2
EPOCHS = 40
OPTIM = 'adam'
LOSS = 'mean_squared_error'

def LSTM_v1(neuro, drop, loss, optim, step, feat):
  model = Sequential()
  model.add(LSTM(neuro, return_sequences=False, batch_input_shape=(None, step, feat)))
  model.add(Dropout(drop))
  model.add(Dense(1, activation='linear'))
  model.compile(loss=loss, optimizer=optim)
  return model

MODEL = LSTM_v1(LSTM_NEURONS, DROPOUT, LOSS, OPTIM, n_steps, n_feat)

MODEL.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=EPOCHS, verbose=1)
MODEL.summary()

#--------
# Plotting the Result
date = stock.index[stock.shape[0]-yTest.shape[0]:stock.shape[0]]
act = List_of_scaler['close'].inverse_transform(yTest.reshape(-1,1)).ravel()
pred = List_of_scaler['close'].inverse_transform(MODEL.predict(xTest)).ravel()

import matplotlib.pyplot as plt
from pandas import DataFrame

results1 = DataFrame({'Actual': act, 'Predicted': pred}, index=date)

results1.plot()
plt.legend(loc='lower right')
plt.xlabel("Dates")
plt.ylabel("Prices")
plt.show()
