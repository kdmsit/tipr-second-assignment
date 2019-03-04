import glob
from PIL import Image
import numpy as np
from random import shuffle
import nn
import kerasnn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import datetime


if __name__ == '__main__':
    path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-2/tipr-second-assignment"
    datasetname="MNIST"
    #datasetname = "Cat-Dog"
    #datasetname = "Dolphins"
    #datasetname = "Pubmed"
    outputpath = "/output/"
    outputFileName = datasetname+"_stat_" + str(datetime.datetime.now()) + ".txt"
    f = open(path + outputpath + outputFileName, "w")
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
            inputPath = "/data/"+datasetname+"/" + str(i) + "/*jpg"
            imlist = []
            for file in glob.glob(path + inputPath):
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
        pca = PCA(n_components=1000).fit(imagePixelList)
        reducedimagePixelList = pca.transform(imagePixelList)
        traindata, testdata, trainlabel, testlabel = train_test_split(reducedimagePixelList, imageLabelList,
                                                                      test_size=0.1, random_state=42)
    elif(datasetname=="Dolphins"):
        inputFilePath = '/data/dolphins/'
        inputFileName = 'dolphins.csv'
        inputLabelFileName = 'dolphins_label.csv'
        #filepath=path+inputFilePath+inputFilePath
        #imagePixelList=pd.read_csv(filepath, sep=',', header=None)
        imagePixelList = np.genfromtxt(inputFilePath+inputFileName, delimiter=' ')
        imageLabelList = np.genfromtxt(inputFilePath+inputLabelFileName, delimiter=' ')
        traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList,test_size=0.1, random_state=42)
    elif (datasetname == "Pubmed"):
        inputFilePath = '/data/pubmed/'
        inputFileName = 'pubmed.csv'
        inputLabelFileName = 'pubmed_label.csv'
        # filepath=path+inputFilePath+inputFilePath
        # imagePixelList=pd.read_csv(filepath, sep=',', header=None)
        imagePixelList = np.genfromtxt(inputFilePath + inputFileName, delimiter=' ')
        imageLabelList = np.genfromtxt(inputFilePath + inputLabelFileName, delimiter=' ')
        traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList, test_size=0.1,
                                                                      random_state=42)
    X = np.asarray(traindata)
    Y = np.asarray(trainlabel)
    Y = Y.reshape((Y.size, 1))
    X_test = np.asarray(testdata)
    Y_test = np.asarray(testlabel)
    Y_test = Y_test.reshape((Y.size, 1))
    kerasnn.MLP(X,Y,X_test,Y_test)

    # region My Custom Code
    '''model={},
    weights={}
    configList=[[600, 50],[500,50],[700,50],[400,50],[600,100],[500,100],[600,100,20],[500,50,20]]    #MNIST
    #configList = [[1000], [500], [700, 50], [500, 50], [600, 100, 20], [500, 50, 20]]                 #Cat-Dog
    #configList = [[100],[60],[100,50],[60,20],[100,50,10]]                                             #Pubmed
    #configList = [[50],[50,10],[50,30,10]]                                                            #Pubmed
    for config in configList:
        print("Configuration Details :",str(config))
        f.write("Configuration Details :" + str(config))
        f.write("\n")
        learning_rate_list = [0.001,0.003,0.005,0.007,0.009,0.01,0.03,0.05,0.07,0.09]  #MNIST
        #learning_rate_list = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]    #Cat-Dog
        #learning_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Dolphin
        #learning_rate_list = [0.001, 0.003, 0.005,0.007,0.009,0.01,0.03,0.05,0.07,0.09]
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
        epoc = 1000
        batchsize = 500
        #batchsize = 10  #Dolphin
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
            accuracy=nn.predict(X_test,testlabel,weights)
            print("Test Accuracy ",accuracy*100)
            f.write("Test Accuracy "+str(accuracy*100))
            f.write("\n")'''
    # endregion



    f.close()
