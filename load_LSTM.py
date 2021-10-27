from utility import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from numpy import array
from save_LSTM import *

model=create_LSTM_model()

scaler,klines,mini,close,df1=LSTM_preparation()
X_train,y_train,X_test,ytest,train_data,test_data=dati_for_LSTM()


checkpoint_path = "salvataggi/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Loads the weights
model.load_weights(checkpoint_path)


### demonstrate prediction for next 10 days, penso che è nelle 5 ore dopo per noi
num=int(len(test_data)/3)
#print(len(test_data))
val=(len(test_data)-num)
#Sarebbe da  modificare in modo che come input utilizzi più dati
print('Dimensione Test_data={} e dimensione val={}'.format(len(test_data),val))
x_input=test_data[val:].reshape(1,-1)
#dim_input=x_input.shape[1]
#print(x_input.shape[1])

## TEST
look_back=100
train_predict=model.predict(X_train)
train_predict=scaler.inverse_transform(train_predict)
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
test_predict=model.predict(X_test)
test_predict=scaler.inverse_transform(test_predict)
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

#tipo aggiungere     X_train, y_train = create_dataset(train_data, time_step)

y_hat = model.predict(X_test)
y_hat_inverse = scaler.inverse_transform(y_hat)

y2_hat = model.predict(y_hat_inverse)
y2_hat_inverse = scaler.inverse_transform(y2_hat)

plt.figure(figsize = (18,9))
plt.plot(y_hat_inverse, label="Predicted Price", color='red')
plt.plot(y2_hat_inverse, label="Predicted Price", color='red')
plt.title('ETH price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.show()

#plt.figure(figsize = (18,9))
#df_temp=y_train
#np.append(df_temp,y_hat_inverse)
#plt.plot(df_temp)
#plt.show()

#y_test_inverse = scaler.inverse_transform(ytest)
plt.figure(figsize = (18,9))
plt.plot(ytest, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')
plt.plot(y2_hat_inverse, label="Predicted futures", color='red')
plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()

print('Plotta')
#è da aumentare
predizionePlot = numpy.empty_like(df1)
predizionePlot[:, :] = numpy.nan
predizionePlot[len(testPredictPlot):len(testPredictPlot)+len(predizionePlot)] = y_hat_inverse
predizionePlot2 = numpy.empty_like(df1)
predizionePlot2[:, :] = numpy.nan
predizionePlot2[len(predizionePlot):len(predizionePlot)+len(predizionePlot2)] = y2_hat_inverse

plt.figure(figsize = (18,9))
#plt.plot(X_train)
plt.plot(scaler.inverse_transform(df1), label="DF1")
#plt.plot(testPredictPlot, label="test")
plt.plot(predizionePlot, label="predizione")
plt.plot(predizionePlot2, label="Predicted futures", color='red')

plt.title('ETH price prediction')
plt.legend(loc='best')

plt.show()

print('B')
print('ma che cazzo ne so')
print('A')



temp_input=list(x_input)
temp_input=temp_input[0].tolist()
print('Inizio Ciclo per Candele')

lst_output=[]

n_steps=num-1
candele=30
i=0
while(i<candele):
    #print(i)
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps ,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
day_new=np.arange(1,num+1)
day_pred=np.arange(num+1,num+1+candele)

val=(len(df1)-num)

plt.figure(figsize = (18,9))
plt.plot(day_new,scaler.inverse_transform(df1[val:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.show()

plt.figure(figsize = (18,9))
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3)
plt.show()

print('end')











