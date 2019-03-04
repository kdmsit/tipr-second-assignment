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
    datasetList=["Cat-Dog"]   #Include Catdog here
    for datasetname in datasetList:
        print(datasetname)
        outputpath = "../output/"
        outputFileName = datasetname+"Keras_stat_" + str(datetime.datetime.now()) + ".txt"
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
        model={},
        weights={}
        opdim=0
        batchsize=0
        epoc=0
        if(datasetname=="MNIST"):
            configList = [[600], [500], [400], [300], [100]]
            # configList = [[600,100],[600,50],[500,100],[500,50],[400,100],[400,50],[200,50],[100,20]]
            # configList = [[600, 100, 20],[500, 100, 20],[600, 50, 20],[500, 50, 20],[200,50,20],[100,50,20]]
            opdim = 10
            batchsize = 500
            epoc = 50
        elif (datasetname == "Cat-Dog"):
            configList = [[1000], [800], [400], [200], [100], [500]]
            # configList = [[1000,100], [800,100], [400,100], [200,50], [100,50], [500,100]]
            # configList = [[1000, 100,20], [800, 100,20], [400, 100,20], [200, 50,10], [100, 50,10], [500, 100,20]]
            opdim = 2
            batchsize = 500
            epoc = 50
        for config in configList:
            X = np.asarray(traindata)
            y = []
            for i in range(len(trainlabel)):
                labellist = [0 for i in range(opdim)]
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

            scores=kerasnn.MLP(X,Y,X_test,Y_test,opdim,batchsize,epoc,config)

            print("Test Accuracy ", scores[1]*100)
            f.write("Test Accuracy " + str(scores[1]*100))
            f.write("\n")
            f.close()
