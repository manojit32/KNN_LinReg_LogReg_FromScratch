
from __future__ import division
import pandas as pd
import tools as tl
import math

class NaiveBayes(object):
	def __init__(self, train=None, trainLabel=None):
		self.train = pd.DataFrame(train)
		self.labels = list(set(trainLabel))
		col = len(self.train.columns)
		self.train.insert(col, col, trainLabel)
		self. class_group = self.train.groupby(self.train.iloc[:, -1])
		self.mean = self. class_group.mean()
		self.variance  = self. class_group.var()
	
	def nd_calculate(self, val, mean, variance):
		coff = 1 / (math.sqrt(2 * math.pi * variance))
		exp  = math.exp(- pow(val - mean, 2) / (2 * variance))
		res  = coff * exp
		return res

	def classification(self, trainE):
		groupNum = self. class_group.count()
		groupNumLabel = groupNum.iloc[:, -1].tolist() 
		groupProbility = [n / sum(groupNumLabel) for n in groupNumLabel]
		for i in range(len(trainE)):
			P = []
			for j in range(len(self.labels)):
				P.append(self.nd_calculate(trainE[i], self.mean.iloc[j, i], self.variance.iloc[j, i]))
			groupProbility = [groupProbility[a] * P[a] for a in range(len(P))]
		maxProb = groupProbility.index(max(groupProbility))
		return self.labels[maxProb]

	def prediction(self, testY):
		predictionLabel = []
		for testSample in testY:
			predictionLabel.append(self.classification(testSample))
		return predictionLabel

def NaiveBayesModelMain():
	train, trainLabel, test, testLabel = tl.createDataSet()
	NaiveBayesModel = NaiveBayes(train, trainLabel)
	predictionLabel = NaiveBayesModel.prediction(test)
	acc = tl.accuracy(predictionLabel, testLabel)
	print('NaiveBayesModel Accuracy : ' + str(acc))
	#print('NaiveBayesModel Recall   : ' + str(rec))
	#print('NaiveBayesModel F-value  : ' + str(F))

if __name__ == '__main__':
	NaiveBayesModelMain()

