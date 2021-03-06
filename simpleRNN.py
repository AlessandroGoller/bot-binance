# https://towardsdatascience.com/stock-prediction-using-recurrent-neural-networks-c03637437578
#https://github.com/krishnaik06/Stock-MArket-Forecasting/blob/master/Untitled.ipynb
#https://www.youtube.com/watch?v=H6du_pfuznE

from utility import *
from sklearn.preprocessing import MinMaxScaler
import numpy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from numpy import array

scaler,klines,mini,close,df1=LSTM_preparation()
X_train,y_train,X_test,ytest,train_data,test_data=dati_for_LSTM()
#print(X_train.shape), print(y_train.shape)


## create LSTM
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
print(model.summary())

X_train = X_train.reshape(-1, 100, 1)
X_test  = X_test.reshape(-1, 100, 1)
y_train = y_train.reshape(-1)
ytest = ytest.reshape(-1)

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
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

### demonstrate prediction for next 10 days, penso che ?? nelle 5 ore dopo per noi
print(len(test_data))
val=(len(test_data)-100)
#Sarebbe da  modificare in modo che come input utilizzi pi?? dati
print(val)
x_input=test_data[val:].reshape(1,-1)
#x_input.shape
print(x_input.shape)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()
print('Inizio Ciclo per i giorni')

lst_output=[]

n_steps=100
i=0
while(i<30):
    print(i)
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
    
day_new=np.arange(1,101)
day_pred=np.arange(101,131)

val=(len(df1)-100)

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





