import glob
from PIL import Image
import numpy as np
from random import shuffle
import nn
from sklearn.model_selection import train_test_split
import time



if __name__ == '__main__':
    path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-2/tipr-second-assignment"
    datasetname="MNIST"
    #datasetname = "Cat-Dog"
    outputpath = "/output/"
    outputFileName = datasetname+"_stat_" + str(time.time()) + ".txt"
    f = open(path + outputpath + outputFileName, "w")

    print('Welcome to the world of neural networks!(MNIST Automated Code)')
    f.write('Welcome to the world of neural networks!(MNIST Automated Code)')
    f.write("\n")
    imagePixelList = []
    imageLabelList = []
    imagePixelListTest = []
    imageLabelListTest = []
    if(datasetname=="MNIST"):
        for i in range(0, 10):
            inputPath = "/data/"+datasetname+"/" + str(i) + "/*jpg"
            imlist = []
            for file in glob.glob(path + inputPath):
                imagepix = []
                im = Image.open(file)
                imlist.append(list(im.getdata()))
            for j in range(0, len(imlist)):
                imagePixelList.append(imlist[j])
                imageLabelList.append(i)
    elif(datasetname=="Cat-Dog"):
        count = 0
        dirlist=['cat','dog']
        for i in dirlist:
            inputPath = "/data/"+datasetname+"/" + str(i) + "/*jpg"
            imlist = []
            for file in glob.glob(path + inputPath):
                count=count+1
                if(count==100):
                    break
                imagepix = []
                im = Image.open(file)
                imlist.append(list(im.getdata()))
            for j in range(0, len(imlist)):
                imagePixelList.append(imlist[j])
                if(i=='cat'):
                    imageLabelList.append(0)
                if (i == 'dog'):
                    imageLabelList.append(1)


    traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList, test_size=0.1,
                                                                  random_state=42)
    model={},
    weights={}
    configList=[[600, 50],[500,50],[700,50],[400,50],[600,100],[500,100],[600,100,20],[500,50,20]]
    for config in configList:
        print("Configuration Details :",str(config))
        f.write("Configuration Details :" + str(config))
        f.write("\n")
        learning_rate_list = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]
        # region config Details
        #config = [600, 50]
        ipdim = len(traindata[0])
        opdim = 10
        if (datasetname == "MNIST"):
            opdim = 10
        elif (datasetname == "Cat-Dog"):
            opdim = 1
        hiddendim = config
        layer = hiddendim
        layer.append(opdim)
        layer.insert(0, ipdim)
        # endregion
        epoc = 50
        batchsize = 500
        print("Epoc :", epoc)
        f.write("Epoc :"+ str(epoc))
        f.write("\n")
        f.write("batchsize :"+ str(batchsize))
        f.write("\n")
        for learningrate in learning_rate_list:
            learning_rate = learningrate
            print("Learning Rate :", learning_rate)
            f.write("Learning Rate :"+str(learning_rate))
            f.write("\n")
            for k in range(epoc):
                batchstartIndex=0
                batchendIndex=batchstartIndex+batchsize
                while(batchendIndex <= len(traindata)):
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
                    if(k==0):
                        weights = nn.initialize_parameters(layer)
                    weights = nn.train(model, X, Y,weights,learning_rate)
                    batchstartIndex=batchendIndex
                    batchendIndex=batchstartIndex+batchsize

            X_test = np.asarray(testdata, dtype=None, order=None)
            accuracy=nn.predict(X_test,testlabel,weights)
            print("Test Accuracy ",accuracy*100)
            f.write("Test Accuracy "+str(accuracy*100))
            f.write("\n")
    f.close()
