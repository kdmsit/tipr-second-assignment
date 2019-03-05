import glob
from PIL import Image
import pickle
import numpy as np
from random import shuffle
import nn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import datetime
if __name__ == '__main__':
    path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-2/tipr-second-assignment"
    datasetname="MNIST"
    #datasetname = "Cat-Dog"
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
        traindata, testdata, trainlabel, testlabel = train_test_split(imagePixelList, imageLabelList,test_size=0.9, random_state=42)
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
        pca = PCA(n_components=500).fit(imagePixelList)
        reducedimagePixelList = pca.transform(imagePixelList)
        traindata, testdata, trainlabel, testlabel = train_test_split(reducedimagePixelList, imageLabelList,test_size=0.1, random_state=42)

    weights = {}
    f1 = open('../Pickel/mnist.pkl', 'rb')
    weights = pickle.load(f1)
    f1.close()
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
    '''f = open('mnist.pkl', 'wb')
    pickle.dump(weights, f)
    f.close()'''
    f.close()
