import glob
from PIL import Image
import numpy as np
from random import shuffle
import nn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import datetime
if __name__ == '__main__':
    datasetname="MNIST"
    #datasetname = "Cat-Dog"
    outputpath = "../output/"
    outputFileName = datasetname+"_zero_stat_" + str(datetime.datetime.now()) + ".txt"
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
    if(datasetname=="MNIST"):
        for i in range(0, 10):
            inputPath = "../data/"+datasetname+"/" + str(i) + "/*jpg"
            imlist = []
            for file in glob.glob(inputPath):
                imagepix = []
                im = Image.open(file)
                imlist.append(list(im.getdata()))
            for j in range(0, len(imlist)):
                imagePixelList.append(imlist[j])
                imageLabelList.append(i)
        traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList,test_size=0.1, random_state=42)
    elif(datasetname=="Cat-Dog"):
        dirlist=['cat','dog']
        for i in dirlist:
            inputPath = "../data/"+datasetname+"/" + str(i) + "/*jpg"
            imlist = []
            for file in glob.glob(inputPath):
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
        pca = PCA(n_components=500).fit(imagePixelList)
        reducedimagePixelList = pca.transform(imagePixelList)
        traindata, testdata, trainlabel, testlabel = train_test_split(reducedimagePixelList, imageLabelList,test_size=0.1, random_state=42)

    print(len(traindata))
    print(len(testdata))
    model={},
    weights={}
    configList = [[400]]
    for config in configList:
        print("Configuration Details :",str(config))
        f.write("Configuration Details :" + str(config))
        f.write("\n")
        learning_rate_list = [0.004]  # MNIST
        # region config Details
        #config = [600, 50]
        ipdim = len(traindata[0])
        opdim = 0
        if (datasetname == "MNIST"):
            opdim = 10
        elif (datasetname == "Cat-Dog"):
            opdim = 2
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
