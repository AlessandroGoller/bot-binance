# pylint: disable=C0103
from utility import *

import numpy 
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import os
from tensorflow import keras
import math
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from numpy import array

def LSTM_preparation():
    ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler
    scaler=MinMaxScaler(feature_range=(0,1))

    klines=dati()
    mini,close=suddividi(klines)
    df1=scaler.fit_transform(np.array(close).reshape(-1,1))
    return scaler,klines,mini,close,df1

def dati_for_LSTM():

    #print(df1)
    #plotta_singolo(df1)
    scaler,klines,mini,close,df1=LSTM_preparation()
    ##splitting dataset into train and test split
    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100 #deve essere minore di X_train e X_test
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    X_train = X_train.reshape(-1, 100, 1)
    X_test  = X_test.reshape(-1, 100, 1)
    y_train = y_train.reshape(-1)
    ytest = ytest.reshape(-1)
    return X_train,y_train,X_test,ytest,train_data,test_data

# Define a simple sequential model
def create_LSTM_model():
    ## create LSTM
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    print(model.summary())
    return model

if __name__ == "__main__":
    scaler,klines,mini,close,df1=LSTM_preparation()
    X_train,y_train,X_test,ytest,train_data,test_data=dati_for_LSTM()
    #print(X_train.shape), print(y_train.shape)
    checkpoint_path = "salvataggi/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                   save_weights_only=True,
                                                    verbose=1)
    model=create_LSTM_model()
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=2,batch_size=64,verbose=1,callbacks=[cp_callback])

    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    ### Calculate RMSE performance metrics
    print(math.sqrt(mean_squared_error(y_train,train_predict)))
    print(math.sqrt(mean_squared_error(ytest,test_predict)))
    ### Plotting 
    # shift train predictions for plotting
    look_back=100
    plt.figure(figsize = (18,9))
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()









