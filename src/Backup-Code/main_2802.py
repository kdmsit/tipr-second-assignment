import glob
from PIL import Image
import numpy as np




# region Initialise Parameters
def initHyperParameter(imagePixelList):
    epoch = 1000
    # neural network hyperparameters
    no_of_layers=3
    input_layer_size = len(imagePixelList[1])
    hidden_layer_size = 10
    output_layer_size = 10
    lmbda = 1
    # weight and bias initialization
    weightHidden = np.random.rand(hidden_layer_size,input_layer_size)
    biasHidden = np.random.rand(hidden_layer_size,1)
    weightOut = np.random.rand(hidden_layer_size, output_layer_size)
    biasOut = np.random.rand(output_layer_size,1)
    return input_layer_size,hidden_layer_size,output_layer_size,lmbda,weightHidden,biasHidden,weightOut,biasOut,epoch
# endregion

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
    for i in range(0, 10):
        inputPath = "/data/MNIST/" + str(i) + "/*jpg"
        imlist = []
        for file in glob.glob(path + inputPath):
            im = Image.open(file)
            imlist.append(list(im.getdata()))
        for j in range(0, 100):
            imagePixelList.append(imlist[j])
            imageLabelList.append(i)
    # endregion

    # region Description
    meanVec=np.mean(imagePixelList,axis=0)
    varVec=np.std(imagePixelList,axis=0)
    normalisedPixelList=dataPreprocessing(imagePixelList,meanVec,varVec)
    # endregion

    input_layer_size, hidden_layer_size, output_layer_size, lmbda, weightHidden, biasHidden, weightOut, biasOut,epoch=\
        initHyperParameter(imagePixelList)
    for i in range(epoch):
        testInput=imagePixelList[i]
        testInput=np.reshape(testInput,(784,1))
        input_layer_output=testInput
        hidden_Layer_input=np.dot(weightHidden,input_layer_output)

    print("Hello")