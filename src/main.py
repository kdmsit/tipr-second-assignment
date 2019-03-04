import glob
from PIL import Image
import numpy as np
import kerasnn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import datetime
import datetime

if __name__ == '__main__':
    #path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-2/tipr-second-assignment"
    #datasetname="MNIST"
    #datasetname = "Cat-Dog"
    #datasetname = "Dolphins"
    #datasetname = "Pubmed"
    datasetList=["Dolphins","Pubmed","MNIST","Cat-Dog"]
    for datasetname in datasetList:
        print(datasetname)
        outputpath = "../output/"
        outputFileName = datasetname+"_stat_" + str(datetime.datetime.now()) + ".txt"
        f = open(outputpath + outputFileName, "w")
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

        if (datasetname == "MNIST"):
            for i in range(0, 10):
                inputPath = "../data/" + datasetname + "/" + str(i) + "/*jpg"
                imlist = []
                for file in glob.glob(inputPath):
                    imagepix = []
                    im = Image.open(file)
                    imlist.append(list(im.getdata()))
                for j in range(0, len(imlist)):
                    imagePixelList.append(imlist[j])
                    imageLabelList.append(i)
            traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList,test_size=0.1, random_state=42)
        elif (datasetname == "Cat-Dog"):
            dirlist = ['cat', 'dog']
            for i in dirlist:
                inputPath = "../data/" + datasetname + "/" + str(i) + "/*jpg"
                imlist = []
                for file in glob.glob(inputPath):
                    imagepix = []
                    im = Image.open(file)
                    im = im.convert('1')
                    imlist.append(list(im.getdata()))
                for j in range(0, len(imlist)):
                    imagePixelList.append(imlist[j])
                    if (i == 'cat'):
                        imageLabelList.append(0)
                    if (i == 'dog'):
                        imageLabelList.append(1)
            pca = PCA(n_components=500).fit(imagePixelList)
            reducedimagePixelList = pca.transform(imagePixelList)
            traindata, testdata, trainlabel, testlabel = train_test_split(reducedimagePixelList, imageLabelList,test_size=0.1, random_state=42)
        elif (datasetname == "Dolphins"):
            inputFilePath = '../data/dolphins/'
            inputFileName = 'dolphins.csv'
            inputLabelFileName = 'dolphins_label.csv'
            imagePixelList = np.genfromtxt(inputFilePath + inputFileName, delimiter=' ')
            imageLabelList = np.genfromtxt(inputFilePath + inputLabelFileName, delimiter=' ')
            traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList, test_size=0.1,random_state=42)
        elif (datasetname == "Pubmed"):
            inputFilePath = '../data/pubmed/'
            inputFileName = 'pubmed.csv'
            inputLabelFileName = 'pubmed_label.csv'
            imagePixelList = np.genfromtxt(inputFilePath + inputFileName, delimiter=' ')
            imageLabelList = np.genfromtxt(inputFilePath + inputLabelFileName, delimiter=' ')
            traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList, test_size=0.1,random_state=42)
        model={},
        weights={}
        opdim=0
        batchsize=0
        epoc=0
        if(datasetname=="MNIST"):
            configList = [[600, 50], [500, 50], [700, 50], [400, 50], [600, 100], [500, 100], [600, 100, 20], [500, 50, 20]]
            opdim = 10
            batchsize = 500
            epoc = 50
        elif (datasetname == "Cat-Dog"):
            configList = [[1000], [500], [700, 50], [500, 50], [600, 100, 20], [500, 50, 20]]
            opdim = 2
            batchsize = 500
            epoc = 50
        elif (datasetname == "Dolphins"):
            configList = [[100], [60], [100, 50], [60, 20], [100, 50, 10]]
            opdim = 4
            batchsize = 2
            epoc = 50
        elif (datasetname == "Pubmed"):
            configList = [[50], [50, 10], [50, 30, 10]]
            opdim = 3
            batchsize = 500
            epoc = 50
        print(opdim)
        print(batchsize)
        print(epoc)
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
            labellist = [0 for i in range(opdim)]
            labellist[int(testlabel[i])] = 1
            y_test.append(labellist)
        Y_test = np.asarray(y_test)

        scores=kerasnn.MLP(X,Y,X_test,Y_test,opdim,batchsize,epoc)

        print("Test Accuracy ", scores[1]*100)
        f.write("Test Accuracy " + str(scores[1]*100))
        f.write("\n")
        f.close()
