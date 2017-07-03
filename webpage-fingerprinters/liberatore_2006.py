# This is part of a Python framework to compliment "Robust Smartphone App Identification Via Encrypted Network Traffic Analysis".
# Copyright (C) 2017  Vincent F. Taylor and Riccardo Spolaor
# See LICENSE for more details.
########################
#This Python code was freely readapted from the code for "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail" and redistributed under the same license (GNU General Public License).
#Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
#https://github.com/kpdyer/website-fingerprinting/
########################

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing.data import MinMaxScaler


class JaccardClassifier:
    @staticmethod
    def traceToInstance(trace, class_):
        instance = {}
        for p in trace:
            instance[p] = 1

        instance['class'] = class_
        return instance

    @staticmethod
    def classify(trainingSet, training_class_counter, testingSet):
        bagOfLengths = {}
        for instance in trainingSet:
            if not bagOfLengths.get(instance['class']):
                bagOfLengths[instance['class']] = {}
            for attribute in instance:
                if attribute != 'class':
                    if not bagOfLengths[instance['class']].get(attribute):
                        bagOfLengths[instance['class']][attribute] = 0
                    bagOfLengths[instance['class']][attribute] += 1

        for className in bagOfLengths:
            for length in bagOfLengths[className].keys():
                if bagOfLengths[className][length] < (training_class_counter[className] / 2.0):
                    del bagOfLengths[className][length]

        correctlyClassified = 0
        debugInfo = []

        y_test=[]
        y_predictions=[]
        for instance in testingSet:
            guess = JaccardClassifier.doClassify(bagOfLengths, instance)
            y_test.append(instance['class'])
            y_predictions.append(guess)

        return y_test, y_predictions

    @staticmethod
    def doClassify(bagOfLengths, instance):
        guess = None
        bestSimilarity = 0
        for className in bagOfLengths:
            intersection = 0
            for attribute in instance:
                if attribute != 'class' and attribute in bagOfLengths[className]:
                    intersection += 1
            union = (len(instance) - 1) + len(bagOfLengths[className])
            if union == 0:
                similarity = 0
            else:
                similarity = 1.0 * intersection / union
            if guess == None or similarity > bestSimilarity:
                bestSimilarity = similarity
                guess = className

        return guess

    @staticmethod
    def liberatore_dataset(X_data, y_data):
        instances = []

        class_counter = {}
        for i, lab_ in enumerate(y_data):
            class_counter[lab_] = class_counter[lab_] + 1 if lab_ in class_counter else 1
            instances.append( JaccardClassifier.traceToInstance(X_data[i], lab_))
        return instances, class_counter


def tracehistogram(trace, noDict=True, direction=True):
    histogram_ = {}
    for p in trace:
        key = "%s" % p if direction is True else np.abs(p)
        if not histogram_.get(key):
            histogram_[key] = 1
        else:
            histogram_[key] += 1

    if noDict is True:
        hist_no_dict = []
        l = -1516 if direction is True else 0
        for i in range(l, 1515, 1):
            hist_no_dict.append(histogram_["%s" % i] if "%s" % i in histogram_ else 0)
        return hist_no_dict
    else:
        return histogram_


class LiberatoreClassifierNB:
    @staticmethod
    def traceToInstance(trace_, direction=True):
        histogram_ = {}
        for p in trace_:
            key = "%s" % p if direction is True else np.abs(p)
            if not histogram_.get(key):
                histogram_[key] = 1
            else:
                histogram_[key] += 1

        hist_no_dict = []
        l = -1516 if direction is True else 0
        for i in range(l, 1515, 1):
            hist_no_dict.append(histogram_["%s" % i] if "%s" % i in histogram_ else 0)
        return hist_no_dict

    @staticmethod
    def liberatore_dataset(X_data, y_data):
        instances = []

        for i, lab_ in enumerate(y_data):
            instances.append( LiberatoreClassifierNB.traceToInstance(X_data[i]))
        return instances, y_data

    @staticmethod
    def classify(X_train, y_train, X_test, y_test):

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)

        liberatore_NB = GaussianNB()

        liberatore_NB.fit(X_train, y_train)
        del X_train

        X_test = scaler.transform(X_test)

        predictions = liberatore_NB.predict(X_test)

        return y_test, predictions

def classifier_liberatore_jaccard(X_train_, y_train_, X_test_, y_test_):


    trainingSet, class_counter = JaccardClassifier.liberatore_dataset(X_train_, y_train_)

    #print(X_train[8])
    testSet,_ = JaccardClassifier.liberatore_dataset(X_test_, y_test_)

    y_test, predictions = JaccardClassifier.classify(trainingSet, class_counter, testSet)

    return y_test, predictions

def classifier_liberatore_NB(X_train_, y_train_, X_test_, y_test_):


    X_train, y_train = LiberatoreClassifierNB.liberatore_dataset(X_train_, y_train_)

    #print(X_train[8])
    X_test, y_test = LiberatoreClassifierNB.liberatore_dataset(X_test_, y_test_)

    y_test, predictions = LiberatoreClassifierNB.classify(X_train, y_train, X_test, y_test)

    return y_test, predictions
