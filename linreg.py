from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
admission_data = pd.read_csv("AdmissionDataset/data.csv")
X = admission_data[["GRE Score","TOEFL Score","University Rating","SOP","LOR ","CGPA","Research"]]
Y = admission_data["Chance of Admit "]
training_X,validation_X,training_Y,validation_Y = train_test_split(X, Y, test_size=0.20,random_state=100)
class linear_regression:
    def __init__(self, bias=False):
        self.bias = bias
        self.coefficient = None

    def fit(self, X, Y):
        if self.bias:
            X = np.insert(np.array(X), X.shape[1], 1, axis=1)
        a = np.dot(X.T, X)
        b = np.dot(X.T, Y)
        a_invese = np.linalg.inv(a)
        self.coefficient = np.dot(a_invese, b)

    def predict(self, X):
        if self.bias:
            X = np.insert(np.array(X), X.shape[1], 1, axis=1)
        return np.dot(X, self.coefficient)

    def RSS(self, X, Y):
        if self.bias:
            X = np.insert(np.array(X), X.shape[1], 1, axis=1)
        predicted = np.dot(X, self.coefficient)
        square_diff = (validation_Y - predicted) ** 2
        u= np.sum(square_diff)
        tot=(validation_Y - validation_Y.mean()) ** 2
        v=np.sum(tot)
        return (1-u/v)
    
    def RSA(self, X, Y):
        if self.bias:
            X = np.insert(np.array(X), X.shape[1], 1, axis=1)
        predicted = np.dot(X, self.coefficient)
        square_diff = abs(validation_Y - predicted)
        return np.sum(square_diff)
    
    def mean(self, X, Y):
        if self.bias:
            X = np.insert(np.array(X), X.shape[1], 1, axis=1)
        predicted = np.dot(X, self.coefficient)
        square_diff = (Y - predicted) ** 2
        u= np.sum(square_diff)
        tot=(validation_Y - validation_Y.mean()) ** 2
        v=np.sum(tot)
        return u/len(Y)


def run():
    
    admission_classifier = linear_regression(bias=True)
    admission_classifier.fit(training_X,training_Y)
    rss = admission_classifier.RSS(validation_X, validation_Y)
    print("The R2 Score of the validation set is %f" % rss)
    mean = admission_classifier.mean(validation_X, validation_Y)
    print("The MSE of the validation set is %f" % mean)
    predicted = admission_classifier.predict(validation_X)
    coefficients = plt.figure(1)
    plt.title("Regression coefficients")
    indices = range(len(admission_classifier.coefficient)-1)
    plt.plot(indices, admission_classifier.coefficient[0:-1], linestyle='-', marker='o')
    plt.xlabel('Indices', fontsize=14, color='blue')
    plt.ylabel('Coefficient', fontsize=14, color='blue')
    plt.grid(True)
    print(admission_classifier.coefficient[0:-1])
    
def run2():
    
    admission_classifier = linear_regression(bias=True)
    admission_classifier.fit(training_X,training_Y)
    rss = admission_classifier.RSS(validation_X, validation_Y)
    print("The R2 Score of the validation set is %f" % rss)
    mean = admission_classifier.mean(validation_X, validation_Y)
    print("The MSE of the validation set is %f" % mean)
    predicted = admission_classifier.predict(validation_X)
    residual=predicted-validation_Y
    plt.scatter(validation_X["CGPA"],residual)
    plt.xlabel("CGPA")
    plt.ylabel("Residual")
    plt.show()
    plt.scatter(validation_X["Research"],residual)
    plt.xlabel("Research")
    plt.ylabel("Residual")
    plt.show()
        