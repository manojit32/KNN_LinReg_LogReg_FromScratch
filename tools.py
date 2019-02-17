
from __future__ import division
import pandas as pd
import numpy as np

def createDataSet(splitSize=0.2):
    d = pd.read_csv('LoanDataset/data.csv')
    trainData = d.iloc[:,[1,2,3,5,6,7,8,10,9]]
    numpyTrainData = np.array(trainData)
    recordNums = numpyTrainData.shape[0]
    trainDataIndex = list(range(recordNums))
    testDataIndex = []
    testNumber = int(recordNums * splitSize)
    for i in range(testNumber):
    	randomNum = int(np.random.uniform(0, len(trainDataIndex)))
    	testDataIndex.append(trainDataIndex[randomNum])
    	trainDataIndex.remove(trainDataIndex[randomNum])
    trainSet = numpyTrainData[trainDataIndex]
    testSet  = numpyTrainData[testDataIndex]
    trainSet = trainSet.tolist()
    testSet  = testSet.tolist()

    trainLabel = [a[-1]  for a in trainSet]
    trainSet   = [a[:-1] for a in trainSet]
    testlabel  = [a[-1]  for a in testSet]
    testSet    = [a[:-1] for a in testSet]
    return trainSet, trainLabel, testSet, testlabel

from sklearn.metrics import classification_report,confusion_matrix
def accuracy(predictionLabel, testLabel):
    cnt = 0
    for i in range(len(testLabel)):
        #print(predictionLabel[i],testLabel[i])
        if predictionLabel[i] == testLabel[i]:
            cnt += 1    
    acc = cnt / len(testLabel)
    print(confusion_matrix(predictionLabel,testLabel))
    print(classification_report(predictionLabel,testLabel))
    return acc


    



