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
    outputpath="/output/"
    outputFileName = "MNIST_stat_"+str(time.time())+".txt"
    f = open(path+outputpath+outputFileName, "w")
    print('Welcome to the world of neural networks!')
    f.write('Welcome to the world of neural networks!')
    f.write("\n")
    filecount={}
    for i in range(0, 10):
        inputPath = "/data/MNIST/" + str(i) + "/*jpg"
        a=glob.glob(path + inputPath)
        filecount[str(i)]=len(a)

    imagePixelList = []
    imageLabelList = []
    imagePixelListTest = []
    imageLabelListTest = []
    for i in range(0, 10):
        inputPath = "/data/MNIST/" + str(i) + "/*jpg"
        imlist = []
        for file in glob.glob(path + inputPath):
            imagepix=[]
            im = Image.open(file)
            imlist.append(list(im.getdata()))
        for j in range(0, len(imlist)):
            imagePixelList.append(imlist[j])
            imageLabelList.append(i)
    traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList, test_size = 0.05, random_state = 42)



    model={}
    #learning_rate_list = [0.01,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    learning_rate=0.03
    epoclist=[1000,2000,3000,4000,5000,6000,7000,8000,9000]

    batchsize = 1000
    for epoc in epoclist:
        print("Epoc :", epoc)
        f.write("Epoc :" + str(epoc))
        f.write("\n")
        print("Learning Rate :",learning_rate)
        f.write("Learning Rate :"+str(learning_rate))
        f.write("\n")
        model = nn.initialize_parameters(traindata[0], 100, 10)
        for epocin in range(epoc):
            print(epocin)
            # region Batch Run
            batchstartIndex=0
            batchendIndex=batchstartIndex+batchsize
            while(batchendIndex <= len(traindata)):
                #print(k,"th Batch Started")
                batchImagePixels=[]
                batchImageLabels = []
                batchImagePixels=[traindata[i] for i in range(batchstartIndex,batchendIndex)]
                batchImageLabels = [trainlabel[i] for i in range(batchstartIndex, batchendIndex)]
                X = np.asarray(batchImagePixels, dtype=None, order=None)
                y = []
                for i in range(len(batchImageLabels)):
                    labellist = [0 for i in range(10)]
                    labellist[int(batchImageLabels[i])] = 1
                    y.append(labellist)
                Y = np.asarray(y, dtype=None, order=None)
                model = nn.train(model, X, Y, epoc, learning_rate)
                batchstartIndex=batchendIndex
                batchendIndex=batchstartIndex+batchsize
            # endregion
        X_test = np.asarray(testdata, dtype=None, order=None)
        accuracy=nn.predict(X_test,testlabel,model)
        print("Test Accuracy ",accuracy*100)
        f.write("Test Accuracy "+str(accuracy*100))
        f.write("\n")
        f.close()

