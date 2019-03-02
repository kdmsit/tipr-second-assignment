import glob
from PIL import Image
import numpy as np
from random import shuffle
import nn
import time

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
    batchsize=1000

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
            imagepix=list(im.getdata())
            imagepix.append(i)
            imlist.append(imagepix)
        for j in range(0, len(imlist)):
            imagePixelList.append(imlist[j])
    shuffle(imagePixelList)
    imageLabelList=[imagePixelList[i][784] for i in range(len(imagePixelList))]
    imagePixel=[]
    imagePixel=[imagePixelList[i][0:784] for i in range(len(imagePixelList))]



    model={}
    learning_rate_list = [0.01,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    epoc = 100
    print("Epoc :", epoc)
    f.write("Epoc :" + str(epoc))
    f.write("\n")
    for learningrate in learning_rate_list:
        learning_rate=learningrate
        print("Learning Rate :",learning_rate)
        f.write("Learning Rate :"+str(learning_rate))
        f.write("\n")
        batchstartIndex=0
        batchendIndex=batchstartIndex+batchsize
        k=1
        while(batchendIndex <= len(imagePixel)):
            #print(k,"th Batch Started")
            batchImagePixels=[]
            batchImageLabels = []
            batchImagePixels=[imagePixel[i] for i in range(batchstartIndex,batchendIndex)]
            batchImageLabels = [imageLabelList[i] for i in range(batchstartIndex, batchendIndex)]
            X = np.asarray(batchImagePixels, dtype=None, order=None)
            y = []
            for i in range(len(batchImageLabels)):
                labellist = [0 for i in range(10)]
                labellist[int(batchImageLabels[i])] = 1
                y.append(labellist)
            Y = np.asarray(y, dtype=None, order=None)
            if(k==1):
                model = nn.initialize_parameters(X, 100, 10)
            model = nn.train(model, X, Y, epoc, learning_rate)
            batchstartIndex=batchendIndex
            batchendIndex=batchstartIndex+batchsize
            k=k+1

        for i in range(0, 10):
            inputPath = "/data/MNIST/" + str(i) + "/*jpg"
            imlist = []
            for file in glob.glob(path + inputPath):
                im = Image.open(file)
                imlist.append(list(im.getdata()))
            for k in range(201, 210):
                imagePixelListTest.append(imlist[k])
                imageLabelListTest.append(i)
        X_test = np.asarray(imagePixelListTest, dtype=None, order=None)
        y_test = []
        for i in range(len(imageLabelListTest)):
            labellist = [0 for i in range(10)]
            labellist[int(imageLabelListTest[i])] = 1
            y.append(labellist)
        Y_test = np.asarray(y_test, dtype=None, order=None)
        accuracy=nn.predict(X_test,imageLabelListTest,model)
        print("Test Accuracy ",accuracy*100)
        f.write("Test Accuracy "+str(accuracy*100))
        f.write("\n")
    f.close()

