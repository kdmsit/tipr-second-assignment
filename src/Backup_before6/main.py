import glob
from PIL import Image
import numpy as np
from random import shuffle
import nn
import time
from sklearn.model_selection import train_test_split

def dataPreprocessing(data,mean,var):
    normalisedData=[]
    for vector in data:
        normalisedVector=np.divide(np.subtract(vector,mean),var)
        normalisedData.append(normalisedVector)
    return normalisedData


if __name__ == '__main__':
    path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-2/tipr-second-assignment"
    imagePixelList = []
    imageLabelList = []
    imagePixelListTest = []
    imageLabelListTest = []
    for i in range(0, 10):
        inputPath = "/data/MNIST/" + str(i) + "/*jpg"
        imlist = []
        for file in glob.glob(path + inputPath):
            imagepix = []
            im = Image.open(file)
            imlist.append(list(im.getdata()))
        for j in range(0, len(imlist)):
            imagePixelList.append(imlist[j])
            imageLabelList.append(i)

    # region Xor-Data
    '''X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])'''

    '''for i in range(2500):
        model = nn.train(X,Y,weights, learning_rate)
    accuracy = nn.predict(X, Y, weights)'''
    # endregion


    # region MNIST Part
    traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList, test_size=0.05,
                                                                  random_state=42)
    model = {}
    learning_rate = 0.3
    config = [1500, 100]
    ipdim = len(traindata[0])
    opdim = 10
    hiddendim = config
    layer = hiddendim
    layer.append(opdim)
    layer.insert(0, ipdim)
    weights = nn.initialize_parameters(layer)
    epochs = 20
    for epoch in range(epochs):
        print("epoch",epoch)
        batchsize=200
        mini_batches_data = [traindata[k:k + batchsize] for k in range(0, len(traindata), batchsize)]
        mini_batches_label = [trainlabel[k:k + batchsize] for k in range(0, len(trainlabel), batchsize)]
        for i in range(len(mini_batches_data)):
            #print("batch", i)
            mini_batch_data=mini_batches_data[i]
            mini_batch_label=mini_batches_label[i]
            X = np.asarray(mini_batch_data, dtype=None, order=None)
            y = []
            for i in range(len(mini_batch_label)):
                labellist = [0 for i in range(10)]
                labellist[int(mini_batch_label[i])] = 1
                y.append(labellist)
            Y = np.asarray(y, dtype=None, order=None)
            model = nn.train(X,Y,weights,learning_rate)
    X_test = np.asarray(testdata, dtype=None, order=None)
    accuracy=nn.predict(X_test,testlabel,model)
    print(accuracy)
    # endregion

