from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def MLP(X, Y,X_test,Y_test):
    np.random.seed(7)
    # create model
    model = Sequential()
    model.add(Dense(600, input_dim=784, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=50, batch_size=500)
    # evaluate the model
    scores = model.evaluate(X_test,Y_test)
    return scores