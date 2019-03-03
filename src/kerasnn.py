from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def MLP(X, Y):
    np.random.seed(7)
    # create model
    model = Sequential()
    model.add(Dense(600, input_dim=785, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=50, batch_size=500)
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print("Hello")
    return