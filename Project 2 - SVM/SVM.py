# Author: Sai Vikhyath
# Date: 07 November 2022


from libsvm.svmutil import *
import numpy as np
import scipy.io



trainData = scipy.io.loadmat("Dataset\\trainData.mat")

trainX1 = np.asarray(trainData["X1"])
trainX2 = np.asarray(trainData["X2"])
trainX3 = np.asarray(trainData["X3"])

trainY = np.asarray(trainData["Y"].flatten())


m1 = svm_train(trainY, trainX1, "-c 10 -t 0")
m2 = svm_train(trainY, trainX2, "-c 10 -t 0")
m3 = svm_train(trainY, trainX3, "-c 10 -t 0")



testData = scipy.io.loadmat("Dataset\\testData.mat")

testX1 = np.asarray(testData["X1"])
testX2 = np.asarray(testData["X2"])
testX3 = np.asarray(testData["X3"])

testY = np.asarray(testData["Y"].flatten())

p_label1, p_acc1, p_val1 = svm_predict(testY, testX1, m1)
p_label2, p_acc2, p_val2 = svm_predict(testY, testX2, m2)
p_label3, p_acc3, p_val3 = svm_predict(testY, testX3, m3)

print(p_acc1)
print(p_acc2)
print(p_acc3)


