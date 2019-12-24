#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import talib


# In[30]:


class DataProcess:
    def __init__(self, tick, start, end):
        self.start = start
        self.end = end
        self.tick = tick
        self.whole_df = web.DataReader(tick, 'yahoo', start, end)

    def divide_data(self, back_start, back_end):
        self.df = self.whole_df.drop(["Adj Close"], axis=1)
        self.df["mid"] = (self.df["High"] + self.df["Low"])/2
        self.backtest = self.df[back_start:back_end]
        return self.df, self.backtest

    def process_data(self):
        self.main_df = self.df.copy()
        self.main_df["return_log"] = self.main_df["mid"].pct_change(1)
        self.main_df["Close"] = talib.DEMA(
            np.array(self.main_df["Close"]), timeperiod=3)
        self.main_df["High"] = talib.DEMA(
            np.array(self.main_df["High"]), timeperiod=3)
        self.main_df["Low"] = talib.DEMA(
            np.array(self.main_df["Low"]), timeperiod=3)
        self.main_df["Open"] = talib.DEMA(
            np.array(self.main_df["Open"]), timeperiod=3)
        self.main_df["dema"] = talib.DEMA(
            np.array(self.main_df["Close"]), timeperiod=30)
        self.main_df["kama"] = talib.KAMA(np.array(
            self.main_df["Close"]), timeperiod=30)  # Kaufman Adaptive Moving Average
        self.main_df["trima"] = talib.TRIMA(
            np.array(self.main_df["Close"]), timeperiod=30)  # Triple exponential
        self.main_df["WMA"] = talib.WMA(
            np.array(self.main_df["Close"]), timeperiod=30)  # Weighted moving average
        self.main_df["adx"] = talib.ADX(np.array(self.main_df["High"]), np.array(
            self.main_df["Low"]), np.array(self.main_df["Close"]), timeperiod=14)
        self.main_df["adxr"] = talib.ADXR(np.array(self.main_df["High"]), np.array(
            self.main_df["Low"]), np.array(self.main_df["Close"]), timeperiod=14)
        self.main_df["apo"] = talib.APO(
            np.array(self.main_df["Close"]), fastperiod=12, slowperiod=26)
        self.main_df["aroondown"], self.main_df["aroonup"] = talib.AROON(
            np.array(self.main_df["High"]), np.array(self.main_df["Low"]), timeperiod=14)
        self.main_df["aroonosc"] = talib.AROONOSC(
            np.array(self.main_df["High"]), np.array(self.main_df["Low"]), timeperiod=14)
        self.main_df["bop"] = talib.BOP(np.array(self.main_df["Open"]), np.array(
            self.main_df["High"]), np.array(self.main_df["Low"]), np.array(self.main_df["Close"]))
        self.main_df["cmo"] = talib.CMO(
            np.array(self.main_df["Close"]), timeperiod=14)
        self.main_df["dx"] = talib.DX(np.array(self.main_df["High"]), np.array(
            self.main_df["Low"]), np.array(self.main_df["Close"]), timeperiod=14)
        self.main_df["mfi"] = talib.MFI(np.array(self.main_df["High"]), np.array(self.main_df["Low"]), np.array(
            self.main_df["Close"]), np.array(self.main_df["Volume"]), timeperiod=14)
        self.main_df["minus_di"] = talib.MINUS_DI(np.array(self.main_df["High"]), np.array(
            self.main_df["Low"]), np.array(self.main_df["Close"]), timeperiod=14)
        self.main_df["minus_dm"] = talib.MINUS_DM(
            np.array(self.main_df["High"]), np.array(self.main_df["Low"]), timeperiod=14)
        self.main_df["plus_di"] = talib.PLUS_DI(np.array(self.main_df["High"]), np.array(
            self.main_df["Low"]), np.array(self.main_df["Close"]), timeperiod=14)
        self.main_df["plus_dm"] = talib.PLUS_DM(
            np.array(self.main_df["High"]), np.array(self.main_df["Low"]), timeperiod=14)
        self.main_df["ppo"] = talib.PPO(
            np.array(self.main_df["Close"]), fastperiod=10, slowperiod=20)
        self.main_df["rsi_14"] = talib.RSI(
            np.array(self.main_df["Close"]), timeperiod=14)
        self.main_df["slowk"], self.main_df["slowd"] = talib.STOCH(np.array(self.main_df["High"]), np.array(self.main_df["Low"]), np.array(
            self.main_df["Close"]), fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        self.main_df["macd"], self.main_df["macdsignal"], self.main_df["macdhist"] = talib.MACD(
            np.array(self.main_df["Close"]), fastperiod=12, slowperiod=26, signalperiod=9)
        self.main_df["cci"] = talib.CCI(np.array(self.main_df["High"]), np.array(
            self.main_df["Low"]), np.array(self.main_df["Close"]), timeperiod=14)
        self.main_df["mom20"] = talib.MOM(
            np.array(self.main_df["Close"]), timeperiod=20)
        self.main_df["mom10"] = talib.MOM(
            np.array(self.main_df["Close"]), timeperiod=10)
        self.main_df["ma20"] = talib.SMA(
            np.array(self.main_df["Close"]), timeperiod=20)
        self.main_df["ma10"] = talib.SMA(
            np.array(self.main_df["Close"]), timeperiod=10)
        self.main_df["roc"] = talib.ROC(
            np.array(self.main_df["Close"]), timeperiod=10)
        self.main_df["ult"] = talib.ULTOSC(np.array(self.main_df["High"]), np.array(
            self.main_df["Low"]), np.array(self.main_df["Close"]), timeperiod1=7, timeperiod2=14, timeperiod3=28)
        self.main_df["will"] = talib.WILLR(np.array(self.main_df["High"]), np.array(
            self.main_df["Low"]), np.array(self.main_df["Close"]), timeperiod=14)
        self.main_df["return_1df"] = self.main_df["return_log"].shift(
            -1)
        self.main_df = self.main_df.dropna().iloc[140:, :]
        return self.main_df

    def prepare_model(self, train_start=None, train_end=9000, test_start=9000, test_end=9200, out_start=9200, out_end=9500, scl=MinMaxScaler(), window=600):
        train_df = self.main_df.iloc[train_start:train_end]
        test_df = self.main_df.iloc[test_start:test_end]
        self.out = self.main_df.iloc[out_start:out_end]
        self.train_data_X = np.array(
            train_df.drop(["return_1df"], axis=1).values)
        self.train_data_y = np.array(train_df["return_1df"].values)

        self.test_data_X = np.array(
            test_df.drop(["return_1df"], axis=1).values)
        self.test_data_y = np.array(test_df["return_1df"].values)
        # Seperate for test data
        self.X_out_sample = np.array(self.out.drop(["return_1df"], axis=1).values)
        self.y_out_sample = np.array(self.out["return_1df"].values)

        smoothing_window_size = window

        scaler_min = scl
        for di in range(0, 9000, smoothing_window_size):
            scaler_min.fit(self.train_data_X[di:di+smoothing_window_size, :])
            self.train_data_X[di:di+smoothing_window_size, :] = scaler_min.transform(
                self.train_data_X[di:di+smoothing_window_size, :])
        if (len(self.train_data_X))%600 != 0:
            self.train_data_X[di:, :] = scaler_min.transform(self.train_data_X[di:, :])

        self.test_data_X = scaler_min.transform(self.test_data_X)
        self.X_out_sample = scaler_min.transform(self.X_out_sample)
        return self.train_data_X, self.train_data_y, self.test_data_X, self.test_data_y, self.X_out_sample, self.y_out_sample, self.out
    

    
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# In[ ]:




