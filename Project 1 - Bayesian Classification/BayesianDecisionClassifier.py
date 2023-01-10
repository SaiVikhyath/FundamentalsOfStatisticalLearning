# Author: Sai Vikhyath
# Date: 30 September, 2022


""" 
Design and Implementation logic:
->  Perform feature extraction. Extract the features, Skewness of the image(Ki) and Dark to Bright ratio(Ri).
->  Perform normalization on the features. Use the formula, {(X - Mean) / (Standard Deviation)}
->  Estimate the Maximum Likelyhood Estimates for the Normal Distribution. Use the formulae, {Mu = sum(X) / N} and {Sigma = sum((X - Mu) ^ 2) / N}
->  Apply Bayesian Decision Theory to classify the images. 
    Use the formula, Prediction = {w1 if (P(w1) * P(w1/x) * P(w1/y)) > (P(w2) * P(w2/x) * P(w2/y)) else w2}
->  Using the computed normalization parameters and Maximum Likelyhood Estimates, apply feature extraction, normalization and bayesian 
    classification on test data.
"""

from prettytable import PrettyTable, ALL
from matplotlib import pyplot
import pandas as pd
import numpy as np
import scipy.io
import math
import sys
import os


trainFile = r"Dataset\\train_data.mat"
testFile = r"Dataset\\test_data.mat"

# These variables are used to normalize test data and use MLE's for prediction on test data.
trainMeanSkewness = 0
trainStandardDeviationSkewness = 0
trainMeanBrightDarkRatio = 0
trainStandardDeviationBrightDarkRatio = 0
trainMleMeanSkewness3, trainMleSigmaSkewness3 = 0, 0
trainMleMeanSkewness7, trainMleSigmaSkewness7 = 0, 0
trainMleMeanBrightDarkRatio3, trainMleSigmaBrightDarkRatio3 = 0, 0
trainMleMeanBrightDarkRatio7, trainMleSigmaBrightDarkRatio7 = 0, 0

flag = 0   # Flag maintained to plot graphs only once.


def loadData(file):
    """ Loads data and label based on the given path and file name."""
    rawData = scipy.io.loadmat(file)
    data = rawData["data"]
    label = rawData["label"][0]
    return data, label


def mean(data):
    """ Compute mean."""
    return sum(data) / len(data)


def standardDeviation(data, meanData):
    """ Compute standard deviation."""
    return (sum([((val - meanData) ** 2) for val in data]) / len(data)) ** 0.5


def featureExtraction(data, T):
    """ Extract Skewness(Ki) and Bright Dark Ratio(Ri) from the data.
        Flatten the image into a single list and then compute skewness and dark to bright ratio.
    """
    skewness = []
    brightDarkRatio = []
    for image in data:
        flattenedImage = list(np.concatenate(image).flat)
        meanImage = mean(flattenedImage)
        stdDevImage = standardDeviation(flattenedImage, meanImage)
        ki = sum([pow(y - meanImage, 3) for y in flattenedImage]) / (len(flattenedImage) * pow(stdDevImage, 3))
        brighter = len([i for i in flattenedImage if i > T])
        darker = len(flattenedImage) - brighter
        ri = brighter/darker
        skewness.append(ki)
        brightDarkRatio.append(ri)
    return skewness, brightDarkRatio


def normalization(skewness, brightDarkRatio):
    """ Normalize training dataset features using the mean and standard deviation of training data."""
    global trainMeanSkewness, trainStandardDeviationSkewness, trainMeanBrightDarkRatio, trainStandardDeviationBrightDarkRatio
    meanSkewness = mean(skewness)
    trainMeanSkewness = meanSkewness
    stdDevSkewness = standardDeviation(skewness, meanSkewness)
    trainStandardDeviationSkewness = stdDevSkewness
    meanBrightDarkRatio = mean(brightDarkRatio)
    trainMeanBrightDarkRatio = meanBrightDarkRatio
    stdDevBrightDarkRatio = standardDeviation(brightDarkRatio, meanBrightDarkRatio)
    trainStandardDeviationBrightDarkRatio = stdDevBrightDarkRatio
    return [[((skewness[i] - meanSkewness) / stdDevSkewness), ((brightDarkRatio[i] - meanBrightDarkRatio) / stdDevBrightDarkRatio)] for i in range(len(skewness))]


def normalizatiionTestData(skewness, brightDarkRatio):
    """ Normalize test dataset features using the mean and standard deviation of the training data."""
    return [[((skewness[i] - trainMeanSkewness) / trainStandardDeviationSkewness), ((brightDarkRatio[i] - trainMeanBrightDarkRatio) / trainStandardDeviationBrightDarkRatio)] for i in range(len(skewness))]


def createDataFrame(features, label):
    """ Create a dataframe of features and class label."""
    for i in range(len(features)):
        features[i].append(label[i])
    dataframe = pd.DataFrame(features, columns=["Skewness", "Bright Dark Ratio", "Class"])
    return dataframe


def MLE(data):
    """ Estimate the parameters for the normal distribution from given data."""
    mean = sum(data) / len(data)
    sigma = sum([((val - mean) ** 2) for val in data]) / len(data)
    return (mean, sigma)


def MaximumLikelyHoodFunction(data, mean, sigma):
    """ Compute the maximum likelyhood function given the optimal parameters for the distribution."""
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * np.exp((-(data - mean) ** 2) / (2 * sigma * sigma))


def BayesianDecisionTheory(data, label, T, P3, P7):
    """ Extract features, maximum likelyhood estimates and apply bayesian decision theory for classification of training data."""
    global trainMleMeanSkewness3, trainMleSigmaSkewness3, trainMleMeanSkewness7, trainMleSigmaSkewness7
    global trainMleMeanBrightDarkRatio3, trainMleSigmaBrightDarkRatio3, trainMleMeanBrightDarkRatio7, trainMleSigmaBrightDarkRatio7
    skewness, brightDarkRatio = featureExtraction(data, T)
    features = normalization(skewness, brightDarkRatio)
    skewness3 = [features[i][0] for i in range(len(label)) if label[i] == 3]
    skewness7 = [features[i][0] for i in range(len(label)) if label[i] == 7]
    brightDarkRatio3 = [features[i][1] for i in range(len(label)) if label[i] == 3]
    brightDarkRatio7 = [features[i][1] for i in range(len(label)) if label[i] == 7]
    mleMeanSkewness3, mleSigmaSkewness3 = MLE(skewness3)
    trainMleMeanSkewness3, trainMleSigmaSkewness3 = mleMeanSkewness3, mleSigmaSkewness3
    mleMeanSkewness7, mleSigmaSkewness7 = MLE(skewness7)
    trainMleMeanSkewness7, trainMleSigmaSkewness7 = mleMeanSkewness7, mleSigmaSkewness7
    mleMeanBrightDarkRatio3, mleSigmaBrightDarkRatio3 = MLE(brightDarkRatio3)
    trainMleMeanBrightDarkRatio3, trainMleSigmaBrightDarkRatio3 = mleMeanBrightDarkRatio3, mleSigmaBrightDarkRatio3
    mleMeanBrightDarkRatio7, mleSigmaBrightDarkRatio7 = MLE(brightDarkRatio7)
    trainMleMeanBrightDarkRatio7, trainMleSigmaBrightDarkRatio7 = mleMeanBrightDarkRatio7, mleSigmaBrightDarkRatio7
    global flag
    if flag == 0:
        plotGraph([round(features[i][0], 1) for i in range(len(label)) if label[i] == 3], "Skewness for Class 3", "Distribution of Skewness for Class 3")
        plotGraph([round(features[i][1], 1) for i in range(len(label)) if label[i] == 3], "Bright Dark Ratio 3", "Distribution of Bright Dark Ratio for Class 3")
        plotGraph([round(features[i][0], 1) for i in range(len(label)) if label[i] == 7], "Skewness for Class 7", "Distribution of Skewness for Class 7")
        plotGraph([round(features[i][1], 1) for i in range(len(label)) if label[i] == 7], "Bright Dark Ratio 7", "Distribution of Bright Dark Ratio for Class 7")
        flag += 1
    print("\n\n")
    print("*" * 150)
    print()
    print("=" * 100)
    print("\t\tMAXIMUM LIKELYHOOD ESTIMATES FOR PARAMETERS: {T = " + str(T) + ", P(3) = " + str(P3) + ", P(7) = " + str(P7) + "}")
    print("=" * 100)
    print("MLE for Skewness when class is 3 : ", mleMeanSkewness3, mleSigmaSkewness3)
    print("MLE for Skewness when class is 7 : ", mleMeanSkewness7, mleSigmaSkewness7)
    print("MLE for Bright Dark Ratio when class is 3 : ", mleMeanBrightDarkRatio3, mleSigmaBrightDarkRatio3)
    print("MLE for Bright Dark Ratio when class is 7 : ", mleMeanBrightDarkRatio7, mleSigmaBrightDarkRatio7)
    print("=" * 100)
    dataframe = createDataFrame(features, label)
    dataframe["P(3/Skewness)"] = MaximumLikelyHoodFunction(dataframe["Skewness"], mleMeanSkewness3, mleSigmaSkewness3)
    dataframe["P(7/Skewness)"] = MaximumLikelyHoodFunction(dataframe["Skewness"], mleMeanSkewness7, mleSigmaSkewness7)
    dataframe["P(3/Bright Dark Ratio)"] = MaximumLikelyHoodFunction(dataframe["Bright Dark Ratio"], mleMeanBrightDarkRatio3, mleSigmaBrightDarkRatio3)
    dataframe["P(7/Bright Dark Ratio)"] = MaximumLikelyHoodFunction(dataframe["Bright Dark Ratio"], mleMeanBrightDarkRatio7, mleSigmaBrightDarkRatio7)
    dataframe["P(3)"] = dataframe["P(3/Skewness)"] * dataframe["P(3/Bright Dark Ratio)"] * P3
    dataframe["P(7)"] = dataframe["P(7/Skewness)"] * dataframe["P(7/Bright Dark Ratio)"] * P7
    dataframe["Prediction"] = np.where(dataframe["P(3)"] > dataframe["P(7)"], 3, 7)
    correctClassification = np.where(dataframe["Prediction"] == dataframe["Class"], 1, 0).sum()
    incorrectClassification = len(dataframe) - correctClassification
    actual3Predicted3 = len(dataframe[(dataframe["Class"] == 3) & (dataframe["Prediction"] == 3)])
    actual7Predicted7 = len(dataframe[(dataframe["Class"] == 7) & (dataframe["Prediction"] == 7)])
    actual3Predicted7 = len(dataframe[(dataframe["Class"] == 3) & (dataframe["Prediction"] == 7)])
    actual7Predicted3 = len(dataframe[(dataframe["Class"] == 7) & (dataframe["Prediction"] == 3)])
    print("\n")
    print("=" * 100)
    print("\tTRAIN DATA CONFUSION MATRIX FOR PARAMETERS: {T = " + str(T) + ", P(3) = " + str(P3) + ", P(7) = " + str(P7) + "}")
    print("=" * 100)
    confusionMatrix = PrettyTable()
    confusionMatrix.hrules=ALL
    confusionMatrix.field_names = ["", "Actual Class 3", "Actual Class 7"]
    confusionMatrix.add_row(["Predicted Class 3", actual3Predicted3, actual7Predicted3])
    confusionMatrix.add_row(["Predicted Class 7", actual3Predicted7, actual7Predicted7])
    print(confusionMatrix)
    print("=" * 100)
    return correctClassification, incorrectClassification


def plotGraph(trainData, label, title):
    """ Plot the bar graph - Feature value v/s Frequency of the feature value"""
    pyplot.xlabel("Feature Value")
    pyplot.ylabel("Frequency of values")
    pyplot.ylim(0, 500)
    pyplot.xticks(np.linspace(-5,7,5))
    pyplot.bar(trainData, height=[trainData.count(i) for i in trainData], label=label)
    pyplot.legend(loc="upper right")
    pyplot.title(title)
    pyplot.show()


def testDataPrediction(data, label, T, P3, P7):
    """ Using the normalization parameters and maximum likelyhood estimates obtained during training, classify the test dataset."""
    global flag
    skewness, brightDarkRatio = featureExtraction(data, T)
    features =  normalizatiionTestData(skewness, brightDarkRatio)
    if flag == 1:
        plotGraph([round(features[i][0], 1) for i in range(len(label)) if label[i] == 3], "Skewness for Class 3", "Distribution of Skewness for Class 3")
        plotGraph([round(features[i][1], 1) for i in range(len(label)) if label[i] == 3], "Bright Dark Ratio 3", "Distribution of Bright Dark Ratio for Class 3")
        plotGraph([round(features[i][0], 1) for i in range(len(label)) if label[i] == 7], "Skewness for Class 7", "Distribution of Skewness for Class 7")
        plotGraph([round(features[i][1], 1) for i in range(len(label)) if label[i] == 7], "Bright Dark Ratio 7", "Distribution of Bright Dark Ratio for Class 7")
        flag += 1
    dataframe = createDataFrame(features, label)
    dataframe["P(3/Skewness)"] = MaximumLikelyHoodFunction(dataframe["Skewness"], trainMleMeanSkewness3, trainMleSigmaSkewness3)
    dataframe["P(7/Skewness)"] = MaximumLikelyHoodFunction(dataframe["Skewness"], trainMleMeanSkewness7, trainMleSigmaSkewness7)
    dataframe["P(3/Bright Dark Ratio)"] = MaximumLikelyHoodFunction(dataframe["Bright Dark Ratio"], trainMleMeanBrightDarkRatio3, trainMleSigmaBrightDarkRatio3)
    dataframe["P(7/Bright Dark Ratio)"] = MaximumLikelyHoodFunction(dataframe["Bright Dark Ratio"], trainMleMeanBrightDarkRatio7, trainMleSigmaBrightDarkRatio7)
    dataframe["P(3)"] = dataframe["P(3/Skewness)"] * dataframe["P(3/Bright Dark Ratio)"] * P3
    dataframe["P(7)"] = dataframe["P(7/Skewness)"] * dataframe["P(7/Bright Dark Ratio)"] * P7
    dataframe["Prediction"] = np.where(dataframe["P(3)"] > dataframe["P(7)"], 3, 7)
    correctClassification = np.where(dataframe["Prediction"] == dataframe["Class"], 1, 0).sum()
    incorrectClassification = len(dataframe) - correctClassification
    actual3Predicted3 = len(dataframe[(dataframe["Class"] == 3) & (dataframe["Prediction"] == 3)])
    actual7Predicted7 = len(dataframe[(dataframe["Class"] == 7) & (dataframe["Prediction"] == 7)])
    actual3Predicted7 = len(dataframe[(dataframe["Class"] == 3) & (dataframe["Prediction"] == 7)])
    actual7Predicted3 = len(dataframe[(dataframe["Class"] == 7) & (dataframe["Prediction"] == 3)])
    print()
    print("=" * 100)
    print("\tTEST DATA CONFUSION MATRIX FOR PARAMETERS: {T = " + str(T) + ", P(3) = " + str(P3) + ", P(7) = " + str(P7) + "}")
    print("=" * 100)
    confusionMatrix = PrettyTable()
    confusionMatrix.hrules=ALL
    confusionMatrix.field_names = ["", "Actual Class 3", "Actual Class 7"]
    confusionMatrix.add_row(["Predicted Class 3", actual3Predicted3, actual7Predicted3])
    confusionMatrix.add_row(["Predicted Class 7", actual3Predicted7, actual7Predicted7])
    print(confusionMatrix)
    print("=" * 100)
    print("\n")
    print("=" * 100)
    print("\tTEST DATA ACCURACY AND ERROR-RATE FOR PARAMETERS: {T = " + str(T) + ", P(3) = " + str(P3) + ", P(7) = " + str(P7) + "}")
    print("=" * 100)
    print("Accuracy : %.5f" %(correctClassification * 100 / (correctClassification + incorrectClassification)) + " %")
    print("Error-Rate : %.5f" %(incorrectClassification / (correctClassification + incorrectClassification)))
    print("=" * 100)


if __name__ == "__main__":
    trainData, trainLabel = loadData(trainFile)
    testData, testLabel = loadData(testFile)

    T, P3, P7 = 150, 0.5, 0.5
    correctClassification, incorrectClassification = BayesianDecisionTheory(trainData, trainLabel, T, P3, P7)
    print("\n")
    print("=" * 100)
    print("\tTRAIN DATA ACCURACY AND ERROR-RATE FOR PARAMETERS: {T = 150, P(3) = 0.5, P(7) = 0.5}")
    print("=" * 100)
    print("Accuracy : %.5f" %(correctClassification * 100 / (correctClassification + incorrectClassification)) + " %")
    print("Error-Rate : %.5f" %(incorrectClassification / (correctClassification + incorrectClassification)))
    print("=" * 100)
    print()
    testDataPrediction(testData, testLabel, T, P3, P7)
    print()
    print("*" * 150)
    print("\n\n")
    
    T, P3, P7 = 150, 0.3, 0.7
    correctClassification, incorrectClassification = BayesianDecisionTheory(trainData, trainLabel, T, P3, P7)
    print("\n")
    print("=" * 100)
    print("\tTRAIN DATA ACCURACY AND ERROR-RATE FOR PARAMETERS: {T = 150, P(3) = 0.3, P(7) = 0.7}")
    print("=" * 100)
    print("Accuracy : %.5f" %(correctClassification * 100 / (correctClassification + incorrectClassification)) + " %")
    print("Error-Rate : %.5f" %(incorrectClassification / (correctClassification + incorrectClassification)))
    print("=" * 100)
    print()
    testDataPrediction(testData, testLabel, T, P3, P7)
    print()
    print("*" * 150)
    print("\n\n")

    T, P3, P7 = 200, 0.5, 0.5
    correctClassification, incorrectClassification = BayesianDecisionTheory(trainData, trainLabel, T, P3, P7)
    print("\n")
    print("=" * 100)
    print("\tTRAIN DATA ACCURACY AND ERROR-RATE FOR PARAMETERS: {T = 200, P(3) = 0.5, P(7) = 0.5}")
    print("=" * 100)
    print("Accuracy : %.5f" %(correctClassification * 100 / (correctClassification + incorrectClassification)) + " %")
    print("Error-Rate : %.5f" %(incorrectClassification / (correctClassification + incorrectClassification)))
    print("=" * 100)
    print()
    testDataPrediction(testData, testLabel, T, P3, P7)
    print()
    print("*" * 150)
    print("\n\n")

    T, P3, P7 = 200, 0.3, 0.7
    correctClassification, incorrectClassification = BayesianDecisionTheory(trainData, trainLabel, T, P3, P7)
    print("\n")
    print("=" * 100)
    print("\tTRAIN DATA ACCURACY AND ERROR-RATE FOR PARAMETERS: {T = 200, P(3) = 0.3, P(7) = 0.7}")
    print("=" * 100)
    print("Accuracy : %.5f" %(correctClassification * 100 / (correctClassification + incorrectClassification)) + " %")
    print("Error-Rate : %.5f" %(incorrectClassification / (correctClassification + incorrectClassification)))
    print("=" * 100)
    print()
    testDataPrediction(testData, testLabel, T, P3, P7)
    print()
    print("*" * 150)
    print("\n\n")

