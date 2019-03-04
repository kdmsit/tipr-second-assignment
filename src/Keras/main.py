import glob
from PIL import Image
import numpy as np
from random import shuffle
import nn
<<<<<<< HEAD
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#import pandas as pd
import datetime
if __name__ == '__main__':
    #path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-2/tipr-second-assignment"
=======
import kerasnn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import datetime


if __name__ == '__main__':
    path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-2/tipr-second-assignment"
>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
    datasetname="MNIST"
    #datasetname = "Cat-Dog"
    #datasetname = "Dolphins"
    #datasetname = "Pubmed"
<<<<<<< HEAD
    outputpath = "../output/"
    outputFileName = datasetname+"_stat_" + str(datetime.datetime.now()) + ".txt"
    f = open(outputpath + outputFileName, "w")
=======
    outputpath = "/output/"
    outputFileName = datasetname+"_stat_" + str(datetime.datetime.now()) + ".txt"
    f = open(path + outputpath + outputFileName, "w")
>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
    Message="Welcome to the world of neural networks!"
    print(Message)
    f.write(Message)
    f.write("\n")
    Message="This MultiLayer Neural Network for Dataset "+datasetname
    print(Message)
    f.write(Message)
    f.write("\n")
    imagePixelList = []
    imageLabelList = []
    imagePixelListTest = []
    imageLabelListTest = []
    if(datasetname=="MNIST"):
        for i in range(0, 10):
<<<<<<< HEAD
            inputPath = "../data/"+datasetname+"/" + str(i) + "/*jpg"
            imlist = []
            for file in glob.glob(inputPath):
=======
            inputPath = "/data/"+datasetname+"/" + str(i) + "/*jpg"
            imlist = []
            for file in glob.glob(path + inputPath):
>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
                imagepix = []
                im = Image.open(file)
                imlist.append(list(im.getdata()))
            for j in range(0, len(imlist)):
                imagePixelList.append(imlist[j])
                imageLabelList.append(i)
        traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList,
                                                                      test_size=0.1, random_state=42)
    elif(datasetname=="Cat-Dog"):
        dirlist=['cat','dog']
        for i in dirlist:
            inputPath = "/data/"+datasetname+"/" + str(i) + "/*jpg"
            imlist = []
            for file in glob.glob(path + inputPath):
            #for k in range(100):
                #file=glob.glob(path + inputPath)[k]
                imagepix = []
                im = Image.open(file)
                im = im.convert('1')
                imlist.append(list(im.getdata()))
            for j in range(0, len(imlist)):
                imagePixelList.append(imlist[j])
                if(i=='cat'):
                    imageLabelList.append(0)
                if (i == 'dog'):
                    imageLabelList.append(1)
<<<<<<< HEAD
        pca = PCA(n_components=500).fit(imagePixelList)
=======
        pca = PCA(n_components=1000).fit(imagePixelList)
>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
        reducedimagePixelList = pca.transform(imagePixelList)
        traindata, testdata, trainlabel, testlabel = train_test_split(reducedimagePixelList, imageLabelList,
                                                                      test_size=0.1, random_state=42)
    elif(datasetname=="Dolphins"):
<<<<<<< HEAD
        inputFilePath = 'data/dolphins/'
=======
        inputFilePath = '/data/dolphins/'
>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
        inputFileName = 'dolphins.csv'
        inputLabelFileName = 'dolphins_label.csv'
        #filepath=path+inputFilePath+inputFilePath
        #imagePixelList=pd.read_csv(filepath, sep=',', header=None)
        imagePixelList = np.genfromtxt(inputFilePath+inputFileName, delimiter=' ')
        imageLabelList = np.genfromtxt(inputFilePath+inputLabelFileName, delimiter=' ')
        traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList,test_size=0.1, random_state=42)
    elif (datasetname == "Pubmed"):
<<<<<<< HEAD
        inputFilePath = 'data/pubmed/'
=======
        inputFilePath = '/data/pubmed/'
>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
        inputFileName = 'pubmed.csv'
        inputLabelFileName = 'pubmed_label.csv'
        # filepath=path+inputFilePath+inputFilePath
        # imagePixelList=pd.read_csv(filepath, sep=',', header=None)
        imagePixelList = np.genfromtxt(inputFilePath + inputFileName, delimiter=' ')
        imageLabelList = np.genfromtxt(inputFilePath + inputLabelFileName, delimiter=' ')
        traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList, test_size=0.1,
                                                                      random_state=42)
<<<<<<< HEAD
    print(len(traindata))
    print(len(testdata))
    model={},
    weights={}
    configList = [[600, 50], [500, 50], [700, 50], [400, 50], [600, 100], [500, 100], [600, 100, 20],[500, 50, 20]]  # MNIST
    # configList = [[1000], [500], [700, 50], [500, 50], [600, 100, 20], [500, 50, 20]]                 #Cat-Dog
    # configList = [[100],[60],[100,50],[60,20],[100,50,10]]                                             #Pubmed
    # configList = [[50],[50,10],[50,30,10]]                                                            #Dolphin
=======
    X = np.asarray(traindata)
    y = []
    for i in range(len(trainlabel)):
        labellist = [0 for i in range(10)]
        labellist[int(trainlabel[i])] = 1
        y.append(labellist)
    Y = np.asarray(y)
    X_test = np.asarray(testdata)
    y_test = []
    for i in range(len(testlabel)):
        labellist = [0 for i in range(10)]
        labellist[int(testlabel[i])] = 1
        y_test.append(labellist)
    Y_test = np.asarray(y_test)

    scores=kerasnn.MLP(X,Y,X_test,Y_test)

    # region My Custom Code
    '''model={},
    weights={}
    configList=[[600, 50],[500,50],[700,50],[400,50],[600,100],[500,100],[600,100,20],[500,50,20]]    #MNIST
    #configList = [[1000], [500], [700, 50], [500, 50], [600, 100, 20], [500, 50, 20]]                 #Cat-Dog
    #configList = [[100],[60],[100,50],[60,20],[100,50,10]]                                             #Pubmed
    #configList = [[50],[50,10],[50,30,10]]                                                            #Pubmed
>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
    for config in configList:
        print("Configuration Details :",str(config))
        f.write("Configuration Details :" + str(config))
        f.write("\n")
<<<<<<< HEAD
        learning_rate_list = [0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09]  # MNIST
        # learning_rate_list = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]    #Cat-Dog
        # learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Dolphin
        # learning_rate_list = [0.001, 0.003, 0.005,0.007,0.009,0.01,0.03,0.05,0.07,0.09] # Pubmed
=======
        learning_rate_list = [0.001,0.003,0.005,0.007,0.009,0.01,0.03,0.05,0.07,0.09]  #MNIST
        #learning_rate_list = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]    #Cat-Dog
        #learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Dolphin
        #learning_rate_list = [0.001, 0.003, 0.005,0.007,0.009,0.01,0.03,0.05,0.07,0.09]
>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
        # region config Details
        #config = [600, 50]
        ipdim = len(traindata[0])
        opdim = 0
        if (datasetname == "MNIST"):
            opdim = 10
        elif (datasetname == "Cat-Dog"):
            opdim = 2
        if (datasetname == "Dolphins"):
            opdim = 4
        if (datasetname == "Pubmed"):
            opdim = 3
        hiddendim = config
        layer = hiddendim
        layer.append(opdim)
        layer.insert(0, ipdim)
        # endregion
<<<<<<< HEAD
        epoc = 50
        batchsize = 500
=======
        epoc = 1000
        batchsize = 500
        #batchsize = 10  #Dolphin
>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
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
                    if (k == 0):
                        weights = nn.initialize_parameters(layer)
                    batchImagePixels=[]
                    batchImageLabels = []
                    batchImagePixels=[traindata[i] for i in range(batchstartIndex,batchendIndex)]
                    batchImageLabels = [trainlabel[i] for i in range(batchstartIndex, batchendIndex)]
                    X = np.asarray(batchImagePixels, dtype=None, order=None)
                    y = []
                    for i in range(len(batchImageLabels)):
                        labellist = [0 for i in range(opdim)]
                        labellist[int(batchImageLabels[i])] = 1
                        y.append(labellist)
                    Y = np.asarray(y, dtype=None, order=None)
                    weights = nn.train(model, X, Y, weights, learning_rate)
                    batchstartIndex=batchendIndex
                    batchendIndex=batchstartIndex+batchsize
            X_test = np.asarray(testdata, dtype=None, order=None)
<<<<<<< HEAD
            accuracyOfMyCode, f1_score_macro, f1_score_micro=nn.predict(X_test,testlabel,weights)
            print("Test Accuracy ",accuracyOfMyCode)
            f.write("Test Accuracy "+str(accuracyOfMyCode))
            f.write("\n")
            print("Test F1 Score(Macro) ", f1_score_macro)
            f.write("Test F1 Score(Macro) " + str(f1_score_macro))
            f.write("\n")
            print("Test F1 Score(Micro) ", f1_score_micro)
            f.write("Test F1 Score(Micro) " + str(f1_score_micro))
            f.write("\n")
=======
            accuracy=nn.predict(X_test,testlabel,weights)
            print("Test Accuracy ",accuracy*100)
            f.write("Test Accuracy "+str(accuracy*100))
            f.write("\n")'''
    # endregion

    print("Test Accuracy ", scores[1]*100)
    f.write("Test Accuracy " + str(scores[1]*100))
    f.write("\n")

>>>>>>> d6411173749e936908a5a0b99d132ccda3956b37
    f.close()
