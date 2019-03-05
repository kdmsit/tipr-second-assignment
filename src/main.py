import glob
from PIL import Image
import pickle
import numpy as np
from random import shuffle
import nn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import datetime
import sys
if __name__ == '__main__':
    data = sys.argv[1]
    mode = 0
    if (data == "--test-data"):
        mode = 0 #simply test
    elif (data == "--train-data"):
        mode = 1 #train and test
    if(mode==0):
        testfilepath = sys.argv[2]
        datasetname = sys.argv[4]
    elif (mode == 1):
        configuration =[]
        trainfilepath=sys.argv[2]
        testfilepath = sys.argv[4]
        datasetname = sys.argv[6]
        #con=sys.argv[8].split('')
        con = sys.argv[8]
        x=[]
        con=con[1:-1]
        con=con.split(',')
        x=[]
        for k in con:
            x.append(int(k))
        configuration=x
    outputpath = "../output/"
    outputFileName = datasetname+"_stat_" + str(datetime.datetime.now()) + ".txt"
    f = open(outputpath + outputFileName, "w")
    Message="Welcome to the world of neural networks!"
    print(Message)
    f.write(Message)
    f.write("\n")
    Message="This MultiLayer Neural Network Script Test for Dataset "+datasetname
    print(Message)
    f.write(Message)
    f.write("\n")
    imagePixelList = []
    imageLabelList = []
    imagePixelListTest = []
    imageLabelListTest = []
    traindata=[]
    trainlabel=[]
    testdata=[]
    testlabel=[]
    weights = {}
    if (mode == 0):
        # region Only Test
        if (datasetname.upper() == "MNIST"):
            # region test Data
            for i in range(0, 10):
                testinputPath = testfilepath + "/" + str(i) + "/*jpg"
                imlist = []
                for file in glob.glob(testinputPath):
                    imagepix = []
                    im = Image.open(file)
                    im = im.convert('1')
                    imlist.append(list(im.getdata()))
                for j in range(0, len(imlist)):
                    imagePixelListTest.append(imlist[j])
                    imageLabelListTest.append(i)
            testdata = imagePixelListTest
            testlabel = imageLabelListTest
            # endregion
            f1 = open('../Pickel/mnist.pkl', 'rb')
            weights = pickle.load(f1)
            f1.close()
        elif (datasetname.upper() == "CAT-DOG"):
            dirlist = ['cat', 'dog']
            # region Test Data
            for i in dirlist:
                testinputPath = testfilepath + "/" + str(i) + "/*jpg"
                imlist = []
                for file in glob.glob(testinputPath):
                    imagepix = []
                    im = Image.open(file)
                    im = im.convert('1')
                    imlist.append(list(im.getdata()))
                for j in range(0, len(imlist)):
                    imagePixelListTest.append(imlist[j])
                    if (i == 'cat'):
                        imageLabelListTest.append(0)
                    if (i == 'dog'):
                        imageLabelListTest.append(1)
            pca = PCA(n_components=1000).fit(imagePixelListTest)
            reducedimagePixelList = pca.transform(imagePixelListTest)
            testdata = reducedimagePixelList
            testlabel = imageLabelListTest
            # endregion
            f1 = open('../Pickel/catdog.pkl', 'rb')
            weights = pickle.load(f1)
            print(np.shape(weights))
            f1.close()
        X_test = np.asarray(testdata, dtype=None, order=None)
        accuracyOfMyCode, f1_score_macro, f1_score_micro = nn.predict(X_test, testlabel, weights)
        print("Test Accuracy ", accuracyOfMyCode)
        f.write("Test Accuracy " + str(accuracyOfMyCode))
        f.write("\n")
        print("Test F1 Score(Macro) ", f1_score_macro)
        f.write("Test F1 Score(Macro) " + str(f1_score_macro))
        f.write("\n")
        print("Test F1 Score(Micro) ", f1_score_micro)
        f.write("Test F1 Score(Micro) " + str(f1_score_micro))
        f.write("\n")
        # endregion
    elif (mode == 1):
        # region Train and Test
        learningrate=0
        opdim = 0
        layer=[]
        if(datasetname.upper()=="MNIST"):
            # region Train Data
            for i in range(0, 10):
                traininputPath = trainfilepath+"/" + str(i) + "/*jpg"
                imlist = []
                for file in glob.glob(traininputPath):
                    imagepix = []
                    im = Image.open(file)
                    imlist.append(list(im.getdata()))
                for j in range(0, len(imlist)):
                    imagePixelList.append(imlist[j])
                    imageLabelList.append(i)
            traindata=imagePixelList
            trainlabel=imageLabelList
            # endregion

            # region test Data
            for i in range(0, 10):
                testinputPath = testfilepath+"/" + str(i) + "/*jpg"
                imlist = []
                for file in glob.glob(testinputPath):
                    imagepix = []
                    im = Image.open(file)
                    imlist.append(list(im.getdata()))
                for j in range(0, len(imlist)):
                    imagePixelListTest.append(imlist[j])
                    imageLabelListTest.append(i)
                testdata=imagePixelListTest
                testlabel=imageLabelListTest
            # endregion
            learningrate=0.004
            opdim = 10
        elif(datasetname.upper()=="CAT-DOG"):
            dirlist=['cat','dog']
            # region Train Data
            for i in dirlist:
                traininputPath = trainfilepath+"/" + str(i) + "/*jpg"
                imlist = []
                for file in glob.glob(traininputPath):
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
            traindata = reducedimagePixelList
            trainlabel = imageLabelList
            # endregion

            # region Test Data
            for i in dirlist:
                testinputPath = testfilepath+"/" + str(i) + "/*jpg"
                imlist = []
                for file in glob.glob(testinputPath):
                    imagepix = []
                    im = Image.open(file)
                    im = im.convert('1')
                    imlist.append(list(im.getdata()))
                for j in range(0, len(imlist)):
                    imagePixelListTest.append(imlist[j])
                    if(i=='cat'):
                        imageLabelListTest.append(0)
                    if (i == 'dog'):
                        imageLabelListTest.append(1)
            pca = PCA(n_components=1000).fit(imagePixelListTest)
            reducedimagePixelListtest = pca.transform(imagePixelListTest)
            testdata = reducedimagePixelListtest
            testlabel = imageLabelListTest
            # endregion
            learningrate=0.003
            opdim = 2
        model={},
        config = configuration
        print("Configuration Details :",str(config))
        f.write("Configuration Details :" + str(config))
        f.write("\n")
        # region config Details
        ipdim = len(traindata[0])
        hiddendim = config
        for dim in hiddendim:
            layer.append(dim)
        #layer.append(hiddendim)
        layer.append(opdim)
        layer.insert(0, ipdim)
        # endregion
        # region Batch and Epoch Details
        epoc = 50
        batchsize = 500
        print("Epoc :", epoc)
        f.write("Epoc :"+ str(epoc))
        f.write("\n")
        f.write("batchsize :"+ str(batchsize))
        f.write("\n")
        learning_rate = learningrate
        print("Learning Rate :", learning_rate)
        f.write("Learning Rate :"+str(learning_rate))
        f.write("\n")
        # endregion
        # region Epoc Batch Run
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
        # endregion
        # endregion
        X_test = np.asarray(testdata, dtype=None, order=None)
        print(np.shape(weights))
        print(np.shape(X_test), np.shape(testlabel))
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
    f.close()
