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
num=1000
#print(len(test_data))
val=(len(test_data)-num)
#Sarebbe da  modificare in modo che come input utilizzi più dati
print('Dimensione Test_data={} e dimensione val={}'.format(len(test_data),val))
x_input=test_data[val:].reshape(1,-1)
#dim_input=x_input.shape[1]
#print(x_input.shape[1])

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











