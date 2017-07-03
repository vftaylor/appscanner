# This is part of a Python framework to compliment "Robust Smartphone App Identification Via Encrypted Network Traffic Analysis".
# Copyright (C) 2017  Vincent F. Taylor and Riccardo Spolaor
# See LICENSE for more details.
########################
#This Python code was freely readapted from the code for "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail" and redistributed under the same license (GNU General Public License).
#Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
#https://github.com/kpdyer/website-fingerprinting/
########################
# Liming Lu, Ee-Chien Chang, and Mun Chan. Websitengerprinting and identi cation using ordered feature sequences. In ESORICS. 2010

from __future__ import print_function
import numpy as np
from pylev import levenshtein


def write(str):
    print(str)
    return


class Lu2010Classifier:
    @staticmethod
    def traceToInstance(trace):
        return trace

    packet_DOWN = -1 #in
    packet_UP = 1 #out

    @staticmethod
    def classify(trainingSet, y_train, testingSet, y_test):
        candidateSequences = {}
        for k_t, trace in enumerate(trainingSet):
            if k_t % 100 == 0:
                print("progress %s" % ((float(k_t) / len(trainingSet)) * 100), "%s / %s" % (k_t, len(trainingSet)))

            for d in ["down", "up"]:
                if not candidateSequences.get(y_train[k_t]):
                    candidateSequences[y_train[k_t]] = {}
                    candidateSequences[y_train[k_t]]["up"] = []
                    candidateSequences[y_train[k_t]]["down"] = []

                candidateSequences[y_train[k_t]][d].append([])
                for p in trace:
                    p_sign = "down" if np.sign(p) == Lu2010Classifier.packet_DOWN else "up"
                    if p_sign == d:
                        if d == "up" and np.abs(p) > 300:
                            candidateSequences[y_train[k_t]][d][-1].append(np.abs(p))
                        elif d == "down" and 300 < np.abs(p) < 1450:
                            candidateSequences[y_train[k_t]][d][-1].append(np.abs(p))

        print("starting lu predictions len %s"%len(testingSet))
        predictions=[]

        for k_s, instance in enumerate(testingSet):
            #actual = y_test[k_s]
            guess = Lu2010Classifier.doClassify(candidateSequences, instance)
            if k_s%10==0:
                print("progress %s"%((float(k_s) / len(testingSet)) * 100), "%s / %s"%(k_s,len(testingSet)))
            predictions.append(guess)

        return y_test, predictions

    @staticmethod
    def doClassify(candidateSequences, x_test):
        guess = None

        targetSequenceUp = []
        targetSequenceDown = []
        for k, p in enumerate(x_test):
            p_sign = "down" if np.sign(p) == Lu2010Classifier.packet_DOWN else "up"
            if p_sign == "up" and np.abs(p) > 300:
                targetSequenceUp.append(np.abs(p))
            elif p_sign == "down" and 300 < np.abs(p) < 1450:
                targetSequenceDown.append(np.abs(p))

        similarity = {}
        for className in candidateSequences:
            if not similarity.get(className):
                similarity[className] = 0
            for direction in ["up", "down"]:
                for i in range(len(candidateSequences[className][direction])):
                    if direction == "up":
                        distance = Lu2010Classifier.levenshtein(targetSequenceUp,
                                                                candidateSequences[className][direction][i])
                        maxLen = max(len(targetSequenceUp), len(candidateSequences[className][direction][i]))
                        if len(targetSequenceUp) == 0 or len(candidateSequences[className][direction][i]) == 0:
                            distance = 1.0
                        else:
                            distance /= 1.0 * maxLen

                        similarity[className] += 0.6 * distance
                    elif direction == "down":
                        distance = Lu2010Classifier.levenshtein(targetSequenceDown,
                                                                candidateSequences[className][direction][i])
                        maxLen = max(len(targetSequenceDown), len(candidateSequences[className][direction][i]))
                        if len(targetSequenceDown) == 0 or len(candidateSequences[className][direction][i]) == 0:
                            distance = 1.0
                        else:
                            distance /= 1.0 * maxLen
                        similarity[className] += 0.4 * distance

        bestSimilarity = 0
        for className in similarity:
            a = similarity[className]
            # b=bestSimilarity[className]
            if guess is None or bestSimilarity <= a:
                bestSimilarity = a
                guess = className

        return guess

    @staticmethod
    # from http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance
    def levenshtein(s1, s2):
        s1 = Lu2010Classifier.encode(s1)
        s2 = Lu2010Classifier.encode(s2)
        return levenshtein(unicode(s1), unicode(s2))

    @staticmethod
    def encode(list):
        strList = []
        for val in list:
            # appVal = config.PACKET_RANGE2.index(val)
            appVal = unichr(val)
            strList.append(appVal)

        return ''.join(strList)


def classifier_lu2010(X_train, y_train, X_test, y_test):
    obj = Lu2010Classifier()

    y_test, y_predictions = obj.classify(X_train, y_train, X_test, y_test)

    return y_test, y_predictions
