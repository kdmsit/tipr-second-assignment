import glob
from PIL import Image
import numpy as np
import random
import nn

def dataPreprocessing(data,mean,var):
    normalisedData=[]
    for vector in data:
        normalisedVector=np.divide(np.subtract(vector,mean),var)
        normalisedData.append(normalisedVector)
    return normalisedData


if __name__ == '__main__':
    print('Welcome to the world of neural networks!')
    path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-2/tipr-second-assignment"
    # The entire code should be able to run from this file!

    # region Fetch MNIST 100 Data of each type
    imagePixelList = []
    imageLabelList = []
    imagePixelListTest = []
    imageLabelListTest = []
    for i in range(0, 10):
        inputPath = "/data/MNIST/" + str(i) + "/*jpg"
        imlist = []
        for file in glob.glob(path + inputPath):
            im = Image.open(file)
            imlist.append(list(im.getdata()))
        for j in range(0, 200):
            imagePixelList.append(imlist[j])
            imageLabelList.append(i)
        for k in range(201, 210):
            imagePixelListTest.append(imlist[k])
            imageLabelListTest.append(i)
    # endregion
    X=np.asarray(imagePixelList, dtype=None, order=None)
    X_test = np.asarray(imagePixelListTest, dtype=None, order=None)
    y=[]
    for i in range(len(imageLabelList)):
        labellist = [0 for i in range(10)]
        labellist[int(imageLabelList[i])]=1
        y.append(labellist)
    Y=np.asarray(y, dtype=None, order=None)
    y_test = []
    for i in range(len(imageLabelListTest)):
        labellist = [0 for i in range(10)]
        labellist[int(imageLabelListTest[i])] = 1
        y.append(labellist)
    Y_test = np.asarray(y_test, dtype=None, order=None)

    #X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    #Y = np.array([[0], [1], [1], [0]])
    learning_rate = 0.6
    #reg_lambda = 0.03
    epoc=2000
    model = nn.initialize_parameters(X,100,10)
    model = nn.train(model, X, Y,epoc,learning_rate)
    nn.predict(X_test,imageLabelListTest,model)
