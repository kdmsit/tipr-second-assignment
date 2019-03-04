from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def MLP(X, Y,X_test,Y_test,opdim,batchsize,epoc):
    np.random.seed(7)
    # create model
    model = Sequential()
    model.add(Dense(600, input_dim=len(X[0]), activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(opdim, activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=epoc, batch_size=batchsize)
    # evaluate the model
    scores = model.evaluate(X_test,Y_test)
    return scores