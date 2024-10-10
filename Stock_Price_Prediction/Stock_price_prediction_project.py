# PROJECT: STOCK PRICE PREDICTION
# PREDICTION OF NIFTY 50 INDEX USING LSTM MODEL

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Here we are considering 5 observations in the past.
LOOK_BACK = 5


# convert an array of values into a matrix of features that are the previous time series values in the past
def reconstruct_data(data_set, n=1):
    x, y = [], []
    for i in range(len(data_set) - n - 1):
        a = data_set[i:(i + n), 0]
        x.append(a)
        y.append(data_set[i + n, 0])
    return np.array(x), np.array(y)


# we want to make sure the results will be the same every time we run the algorithm
# np.random.seed(1)

# load the dataset
data_frame = pd.read_csv('Nifty50_Historical_Data.csv', usecols=[1])
print(data_frame)
# plt.plot(data_frame)
# plt.show()

# we just need the temperature column
data = data_frame.values

# we are dealing with floating-point values
data = data.astype('float32')

# min-max normalization Values are between range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# # split into train and test sets (80% - 20%)
train, test = data[0:int(len(data) * 0.8), :], data[int(len(data) * 0.8):len(data), :]

# create the training data and test data matrix
train_x, train_y = reconstruct_data(train, LOOK_BACK)
test_x, test_y = reconstruct_data(test, LOOK_BACK)

# reshape input to be [numOfSamples, time steps, numOfFeatures]
# time steps is 1 because we want to predict the next value (t+1)
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

print(train_x.shape)

# create the LSTM model
model = Sequential()
# model.add(Conv2D(32, kernel_size=1, activation='tanh', padding='same', input_shape=(3994, 1, LOOK_BACK)))
# model.add(MaxPooling2D(pool_size=1, padding='same'))
# model.add(BatchNormalization())
# model.add(Flatten())
model.add(LSTM(units=216, return_sequences=True, input_shape=(1, LOOK_BACK)))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=16))
model.add(Dropout(0.2))
model.add(Dense(units=1))

print(model.summary())

OPTIMIZER = Adam(learning_rate=0.0005)
# optimize the model with ADAM optimizer
model.compile(loss='mean_squared_error', optimizer=OPTIMIZER)
model.fit(train_x, train_y, epochs=10, batch_size=128, verbose=2)

# The data is min-max normalized. So to return to original values we use inverse_transform()
test_predict = model.predict(test_x)
test_predict = scaler.inverse_transform(test_predict)
test_labels = scaler.inverse_transform([test_y])

test_score = mean_squared_error(test_labels[0], test_predict[:, 0])
print('Score on test set: %.2f MSE' % test_score)

# plot the results (original data + predictions)
test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_x) + 2 * LOOK_BACK + 1:len(data) - 1, :] = test_predict
plt.plot(scaler.inverse_transform(data))
plt.plot(test_predict_plot, color="green")
plt.show()
