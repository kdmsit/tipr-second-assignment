from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def MLP(X,Y,X_test,Y_test,opdim,batchsize,epoc,config):
    np.random.seed(7)
    # create model
    model = Sequential()
    model.add(Dense(config[0], input_dim=len(X[0]), activation='sigmoid'))
    if(len(config)>1):
        for i in range(len(config)):
            if(i>0):
                model.add(Dense(config[i], activation='sigmoid'))
    model.add(Dense(opdim, activation='sigmoid'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=epoc, batch_size=batchsize)
    # evaluate the model
    scores = model.evaluate(X_test,Y_test)
    return scores