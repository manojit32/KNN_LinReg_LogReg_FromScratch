
import numpy as np
import pandas as pd
from random import random, seed


def rmse(Y, Ypred):
    rmse = np.sqrt(sum((Y - Ypred) ** 2) / len(Y))
    return rmse

def r2Score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2





'''def costFunction(X, Y, W):
    N = len(Y)
    C = np.sum((X.dot(W) - Y) ** 2)/(2 * N) 
    return C'''

def costFunction(X, Y, W):
    N = len(Y)
    C = np.sum(abs(X.dot(W) - Y))/(N) 
    return C
'''
def costFunction(X, Y, W):
    N = len(Y)
    C = np.sum(abs(X.dot(W) - Y)/Y)*(100/N) 
    return C'''

def gradientDescent(X, Y, W, alpha, maxNumIterations=50000):

    N = len(Y)
    costHistory=[] 
    wHistory=[] 
    iteration = 0 
    while iteration <maxNumIterations:

        h = X.dot(W)
        loss =np.sign(h - Y)
        #loss=h-Y
        gradient = np.sign(X.T.dot(loss)) / N
        #gradient = np.sign(X.T.dot(loss/Y))*(100/N)
        #gradient=(X.T.dot(loss)) / N
        W = W - alpha * gradient
        cost = costFunction(X, Y, W) 
        costHistory.append(cost)      
        iteration = iteration+1 
        wHistory.append(W)  
    return W, costHistory,wHistory






def showResults(X,Y,W,newW,costHistory,maxNumIterations,wHistory):
    

    inital_cost = costFunction(X, Y, W) 
    Y_pred = X.dot(newW)

    dash = '=' * 80 
    print(dash)
    print("LINEAR REGRESSION USING GRADIENT DESCENT")
    print(dash)
    print("        Initial cost:    {:>12,.4f}.".format(inital_cost))
    print("        # Iterations:    {:>12,.0f}.    ".format(maxNumIterations))
    print("          Final cost:    {:>+12.1f}.".format(costHistory[-1]))
    print("                RMSE:    {:>+12.5f}, R-Squared: {:>+12.5f}".format(rmse(Y, Y_pred),r2Score(Y, Y_pred)))
    print(dash)
    print(newW)


def programBody(data,alpha,maxNumIterations):
    
    numXColumns = data.shape[1]-1 
    W = (numXColumns)*[0]          
    x0 = np.ones(data.shape[0]) 

    X = np.column_stack((x0,data.iloc[:, 1:8].values))  
    Y = np.array(data.iloc[:,-1])             
    newW, costHistory,wHistory = gradientDescent(X, Y, W, alpha,maxNumIterations)

    showResults(X, Y, W, newW, costHistory, maxNumIterations,wHistory) 

    return

from sklearn.preprocessing import StandardScaler


def run():
   
    alpha = 0.00001               
    maxNumIterations = 10000000

    data=pd.read_csv("AdmissionDataset/data.csv")
    
    programBody(data, alpha, maxNumIterations)


if __name__ == '__main__':
    run()