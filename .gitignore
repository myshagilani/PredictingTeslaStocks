import pandas as pd
import numpy as np

dataset_train = pd.read_csv("tslatrain.csv")

training_set = dataset_train.iloc[:, 1:2].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled)

#Creating the timestamp datastructures with the 1 output. could change the 60 hyperparameter
input_train = []
benchmark_train = []
for i in range(60, len(training_set_scaled)):
    input_train.append(training_set_scaled[i-60:i, 0])
    benchmark_train.append(training_set_scaled[i, 0])
input_train, benchmark_train = np.array(input_train), np.array(benchmark_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

#Reshaping
input_train = np.reshape(input_train, (input_train.shape[0], input_train.shape[1], 1))

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (input_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(input_train, benchmark_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results
import matplotlib.pyplot as plt
# Getting the real stock price of 2017
dataset_test = pd.read_csv('tslatest.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset_train['open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
print(inputs)

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

input_test = []
for i in range(60, 83):
    input_test.append(inputs[i-60:i, 0])
input_test = np.array(input_test)
input_test = np.reshape(input_test, (input_test.shape[0], input_test.shape[1], 1))
predicted_stock_price = regressor.predict(input_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Tesla Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Tesla Stock Price')
plt.title('Tesla Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Tesla Stock Price')
plt.legend()
plt.show()
