#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:16:08 2019

@author: Guneykan Ozkaya
"""
import sonnet as snt
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from functools import partial
import talib
from sklearn.metrics import accuracy_score
tf.__version__
start_date = '1970-12-31'
end_date = '2019-04-12'
df = web.DataReader('AAPL', 'yahoo', start_date, end_date)
df = df.drop(["Adj Close"], axis=1)
df["mid"] = (df["High"]+df["Low"])/2
df["return_log"] = df["mid"].pct_change(1)
# Smooth data with Double exponential moving average
df["Close"] = talib.DEMA(np.array(df["Close"]), timeperiod=3)
df["High"] = talib.DEMA(np.array(df["High"]), timeperiod=3)
df["Low"] = talib.DEMA(np.array(df["Low"]), timeperiod=3)
df["Open"] = talib.DEMA(np.array(df["Open"]), timeperiod=3)
# Add tech indicators
df["dema"] = talib.DEMA(np.array(df["Close"]), timeperiod=30) #Double Exponential Moving Average
df["kama"] = talib.KAMA(np.array(df["Close"]), timeperiod=30) # Kaufman Adaptive Moving Average
df["trima"] = talib.TRIMA(np.array(df["Close"]), timeperiod=30) # Triple exponential
df["WMA"] = talib.WMA(np.array(df["Close"]), timeperiod=30) # Weighted moving average
df["adx"] = talib.ADX(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=14) 
df["adxr"] = talib.ADXR(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=14) 
df["apo"] = talib.APO(np.array(df["Close"]), fastperiod=12, slowperiod=26)
df["aroondown"], df["aroonup"] = talib.AROON(np.array(df["High"]), np.array(df["Low"]), timeperiod=14) 
df["aroonosc"] = talib.AROONOSC(np.array(df["High"]), np.array(df["Low"]), timeperiod=14)
df["bop"] = talib.BOP(np.array(df["Open"]), np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"])) 
df["cmo"] = talib.CMO(np.array(df["Close"]), timeperiod=14) 
df["dx"] = talib.DX(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=14) 
df["mfi"] = talib.MFI(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), np.array(df["Volume"]), timeperiod=14) 
df["minus_di"] = talib.MINUS_DI(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=14) 
df["minus_dm"] = talib.MINUS_DM(np.array(df["High"]), np.array(df["Low"]), timeperiod=14)
df["plus_di"] = talib.PLUS_DI(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=14)
df["plus_dm"] = talib.PLUS_DM(np.array(df["High"]), np.array(df["Low"]), timeperiod=14)
df["ppo"] = talib.PPO(np.array(df["Close"]), fastperiod=10, slowperiod=20)
df["rsi"] = talib.RSI(np.array(df["Close"]), timeperiod=14)
df["slowk"], df["slowd"] = talib.STOCH(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(np.array(df["Close"]), fastperiod=12, slowperiod=26, signalperiod=9)
df["cci"] = talib.CCI(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=14)
df["mom20"] = talib.MOM(np.array(df["Close"]), timeperiod=20)
df["mom10"] = talib.MOM(np.array(df["Close"]), timeperiod=10)
df["ma20"] = talib.SMA(np.array(df["Close"]), timeperiod=20)
df["ma10"] = talib.SMA(np.array(df["Close"]), timeperiod=10)
df["roc"] = talib.ROC(np.array(df["Close"]), timeperiod=10)
df["ult"] = talib.ULTOSC(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod1=7, timeperiod2=14, timeperiod3=28)
df["will"] = talib.WILLR(np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), timeperiod=14)
df["return_1df"] = df["return_log"].shift(-1)

df = df.dropna()
# Create a copy of data set for denoising
df_ = df.copy()
df_ = df_.drop(["return_1df"], axis=1)


# Train valid test split
train_df_ = df_.iloc[:9000]
test_df_ = df_.iloc[9000:9200]
train_df = df.iloc[:9000]
test_df = df.iloc[9000:9200]
out_of_sample = df.iloc[9200:9500]
out_of_sample_ = df_.iloc[9200:9500]


# Seperate features and target for training data
train_data_X = np.array(train_df_.values)
train_data_y = np.array(train_df["return_1df"].values)
# Seperate for valid data
test_data_X = np.array(test_df_.values)
test_data_y = np.array(test_df["return_1df"].values)
# Seperate for test data
X_out_sample = np.array(out_of_sample_.values)
y_out_sample = np.array(out_of_sample["return_1df"].values)


# Windowized normalization
smoothing_window_size = 600

scaler_min = MinMaxScaler()
for di in range(0,9000,smoothing_window_size):
    scaler_min.fit(train_data_X[di:di+smoothing_window_size,:])
    train_data_X[di:di+smoothing_window_size,:] = scaler_min.transform(train_data_X[di:di+smoothing_window_size,:])

# Scale features of test data
test_data_X = scaler_min.transform(test_data_X)
X_out_sample = scaler_min.transform(X_out_sample)


X_train = train_data_X
y_train = train_data_y
X_test = test_data_X
y_test = test_data_y
X_out_sample = X_out_sample
y_out_sample = y_out_sample

# Model
n_steps = 20
n_inputs = X_train.shape[1]
num_nodes = 200
n_layers = 3
n_outputs = 1

# Reshape it for LSTM
X_test = X_test.reshape((-1, n_steps, n_inputs))
y_test = y_test.reshape((-1, n_steps, n_outputs))    
X_out_sample = X_out_sample.reshape((-1, n_steps, n_inputs))
y_out_sample = y_out_sample.reshape((-1, n_steps, n_outputs))
tf.reset_default_graph()

# Construction
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])



#cells = [tf.nn.rnn_cell.LSTMCell(num_units=num_nodes, activation=tf.nn.elu, use_peepholes=True) for layer in range(n_layers)]
#multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.LSTMCell(num_units=num_nodes, activation=tf.nn.elu, use_peepholes=True), 
                                              output_size=n_outputs)
#cell = tf.contrib.rnn.OutputProjectionWrapper(multi_layer_cell, output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


learning_rate = 0.001


loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()
saver = tf.train.Saver()


max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty
n_iterations = 100
batch_size = 40
mse_train = []
mse_test = []
mse_out_ = []
#Execution
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            y_batch = y_batch.reshape((-1, n_steps, n_outputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
        mse_ = loss.eval(feed_dict={X: X_test, y: y_test})
        mse_out = loss.eval(feed_dict={X: X_out_sample, y: y_out_sample})
        mse_train.append(mse)
        mse_test.append(mse_)
        mse_out_.append(mse_out)
        print(iteration, "\tMSE:", mse, " MSE_TEST", mse_, " MSE_out", mse_out)
        if mse_ < best_loss: # adding stop loss
            save_path = saver.save(sess, "./my_LSTM_model_V2")
            best_loss = mse_
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break



# Prediction      
with tf.Session() as sess:
    saver.restore(sess, "./my_LSTM_model_V2") 
    X_new = X_test
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    X_new_ = X_out_sample
    y_pred_out = sess.run(outputs, feed_dict={X: X_new_})

# Loss values for train, valid, out of sample
plt.plot(mse_train)
plt.plot(mse_test)
plt.plot(mse_out_)

# Daily Predictions
plt.plot(y_test.reshape((-1, 1)))
plt.plot(y_pred.reshape((-1, 1)))
plt.plot(y_out_sample.reshape((-1, 1)))
plt.plot(y_pred_out.reshape((-1, 1)))

# Out of sample, direction scores
comparison_change = pd.concat([pd.DataFrame(y_out_sample.reshape((-1, 1))), 
                        pd.DataFrame(y_pred_out.reshape((-1, 1)))], axis=1)

# Calculate the pct change    
comparison_change.columns = ["True", "Pred"]
comparison_change["True"][100:].plot()
comparison_change["Pred"][100:].plot()

# Tranform regression predictions to classification
test = []
for a in comparison_change["True"]:
    if a >= 0:
        test.append(1)
    else:
        test.append(0)

pred = []
for a in comparison_change["Pred"]:
    if a >= 0:
        pred.append(1)
    else:
        pred.append(0)

# Classification Report
accuracy_score(test, pred)
print(classification_report(test, pred))
np.mean(pred)
np.mean(test)


# Basic daily rebalancing portfolio, 
raw_prediction = y_pred_out.reshape(-1, 1)

Suggestion = []
percentChange = []

for d in range(len(raw_prediction)):
    percentChange.append(float((raw_prediction[d])))
    
for n in percentChange:
    #print(n)
    if n > 0:
        Suggestion.append(-1) # Sell after a threshhold of increase
    elif n < 0:
        Suggestion.append(1) # Buy when there is a drop
    else:
        Suggestion.append(0)
        
y_out_sample = y_out_sample.reshape(-1, 1)
y_pred_out = y_pred_out.reshape(-1, 1)

PurchaseHistory=[]
BuyPoints =  [[], []]
SellPoints = [[], []]
bought = False
for i in range(1, len(y_out_sample)-1):
    if Suggestion[i] == 1 and not bought:
        PurchaseHistory.append(["Buy: ", y_out_sample[i], i])
        BuyPoints[0].append(y_out_sample[i])
        BuyPoints[1].append(i)
        bought = True
    if Suggestion[i] == -1 and bought:
        PurchaseHistory.append(["Sell:", y_out_sample[i], i])
        SellPoints[0].append(y_out_sample[i])
        SellPoints[1].append(i)        
        bought = False
if len(PurchaseHistory)%2 != 0:
    PurchaseHistory.pop()
    BuyPoints[0].pop()
    BuyPoints[1].pop() 

# Profit calculation
Profit = 0
for purchase in range(1, len(PurchaseHistory)+1, 2):
    MoneyMade = float(PurchaseHistory[purchase][1] - float(PurchaseHistory[purchase-1][1]))
    Profit += MoneyMade
print('%'+ str(round(Profit*100, 2)))

# Buy and Sell points according to predictions
plt.figure(figsize=(20,10))
plt.plot(y_out_sample, color =  'purple', label = "Actual Returns")
plt.plot(y_pred_out, color = 'blue', label = "Predicted Returns")
plt.plot(BuyPoints[1], BuyPoints[0], "go")
plt.plot(SellPoints[1], SellPoints[0], "ro")

plt.title("APPL December 2017 - February 2019 Stock Predictions")
plt.xlabel('Time')
plt.ylabel('Apple Daily Returns')
plt.legend()
plt.show()





