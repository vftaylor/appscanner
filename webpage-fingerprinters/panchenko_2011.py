# This is part of a Python framework to compliment "Robust Smartphone App Identification Via Encrypted Network Traffic Analysis".
# Copyright (C) 2017  Vincent F. Taylor and Riccardo Spolaor
# See LICENSE for more details.
########################
#This Python code was freely readapted from the code for "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail" and redistributed under the same license (GNU General Public License).
#Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
#https://github.com/kpdyer/website-fingerprinting/
########################


from __future__ import print_function
import gzip, cPickle
import numpy as np
from sklearn.svm.classes import SVC
from time import time


def tracecountpacketdirection(trace, direction=1):
    count = 0
    for p in trace:
        if np.sign(p) == direction:
            count += 1
    return count


def tracebandwidthdirection(trace, direction=1, total=False):
    count = 0
    count_ = 0
    for p in trace:
        if np.sign(p) == direction:
            count += np.abs(p)
        else:
            count_ += np.abs(p)
    if total is True:
        return count + count_
    else:
        return count


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


class Panchenko2011Classifier:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x) / base))

    @staticmethod
    def roundNumberMarker(n):
        if n == 4 or n == 5:
            return 3
        elif n == 7 or n == 8:
            return 6
        elif n == 10 or n == 11 or n == 12 or n == 13:
            return 9
        else:
            return n

    def tracestoInstances(self, traces):
        instances = []
        for t in traces:
            instances.append(Panchenko2011Classifier.traceToInstance(t))
        return instances

    @staticmethod
    def traceToInstance(trace_):
        if len(trace_) == 0:
            instance_ = {}
            # instance['class'] = label
            return instance_

        instance_ = tracehistogram(trace_, noDict=False, direction=True)

        # Size/Number Markers
        directionCursor = None
        dataCursor = 0
        numberCursor = 0
        for packet_ in trace_:
            if directionCursor == None:
                directionCursor = np.sign(packet_)  # packet.getDirection()

            if np.sign(packet_) != directionCursor:
                dataKey = "S_%s_%s" % (directionCursor, Panchenko2011Classifier.roundArbitrary(dataCursor, 600))
                if dataKey not in instance_.keys():
                    instance_[dataKey] = 0
                instance_[dataKey] += 1

                numberKey = "N_%s_%s" % (directionCursor, Panchenko2011Classifier.roundNumberMarker(numberCursor))
                if numberKey not in instance_.keys():
                    instance_[numberKey] = 0
                instance_[numberKey] += 1

                directionCursor = np.sign(packet_)  # .getDirection()
                dataCursor = 0
                numberCursor = 0

            dataCursor += np.abs(packet_)
            numberCursor += 1

        if dataCursor > 0:
            key = "S_%s_%s" % (directionCursor, Panchenko2011Classifier.roundArbitrary(dataCursor, 600))
            if not instance_.get(key):
                instance_[key] = 0
            instance_[key] += 1

        if numberCursor > 0:
            numberKey = "N_%s_%s" % (directionCursor, Panchenko2011Classifier.roundNumberMarker(numberCursor))
            if not instance_.get(numberKey):
                instance_[numberKey] = 0
            instance_[numberKey] += 1

        # HTML Markers
        counterUP = 0
        counterDOWN = 0
        htmlMarker = 0
        for packet_ in trace_:
            if np.sign(packet_) == 1:
                counterUP += 1
                if counterUP > 1 and counterDOWN > 0: break
            elif np.sign(packet_) == -1:
                counterDOWN += 1
                htmlMarker += np.abs(packet_)

        htmlMarker = Panchenko2011Classifier.roundArbitrary(htmlMarker, 600)
        instance_['H' + str(htmlMarker)] = 1

        # Ocurring Packet Sizes
        packetsUp = []
        packetsDown = []
        for packet_ in trace_:
            if np.sign(packet_) == 1 and np.abs(packet_) not in packetsUp:
                packetsUp.append(np.abs(packet_))
            if np.sign(packet_) == -1 and np.abs(packet_) not in packetsDown:
                packetsDown.append(np.sign(packet_))
        instance_['uniquePacketSizesUp'] = Panchenko2011Classifier.roundArbitrary(len(packetsUp), 2)
        instance_['uniquePacketSizesDown'] = Panchenko2011Classifier.roundArbitrary(len(packetsDown), 2)

        # Percentage Incoming Packets
        instance_['percentageUp'] = Panchenko2011Classifier.roundArbitrary(
            100.0 * tracecountpacketdirection(trace_, 1) / len(trace_), 5)
        instance_['percentageDown'] = Panchenko2011Classifier.roundArbitrary(
            100.0 * tracecountpacketdirection(trace_, -1) / len(trace_), 5)

        # Number of Packets
        instance_['numberUp'] = Panchenko2011Classifier.roundArbitrary(tracecountpacketdirection(trace_, 1), 15)
        instance_['numberDown'] = Panchenko2011Classifier.roundArbitrary(tracecountpacketdirection(trace_, -1), 15)

        # Total Bytes Transmitted
        bandwidthUp = Panchenko2011Classifier.roundArbitrary(tracebandwidthdirection(trace_, 1), 10000)
        bandwidthDown = Panchenko2011Classifier.roundArbitrary(tracebandwidthdirection(trace_, -1), 10000)
        instance_['0-B' + str(bandwidthUp)] = 1
        instance_['1-B' + str(bandwidthDown)] = 1

        # Label
        # instance['class'] = 'webpage'+str(trace.getId())

        return instance_

    @staticmethod
    def fit(y_train, trainingSet,max_iter=5000):
        svm_ = SVC(
            kernel="rbf",
            C=131072,
            gamma=0.0000019073486328125,
            max_iter=max_iter,
            verbose=5,
            class_weight="balanced"

        )
        #print("start Pachenko fitting...")
        time0 = time()

        svm_.fit(trainingSet, y_train)
        time1 = time()
        #print("completed in %s seconds" % (time1 - time0))
        return svm_

    @staticmethod
    def predict(classif, X_test):
        #print("prediction")
        time1 = time()
        labels = classif.predict(X_test)
        #print("completed in %s seconds" % (time() - time1))
        return labels


def classifier_panchenko2011(X_train, y_train, X_test, y_test, max_iter=5000):
    pac = Panchenko2011Classifier()
    X_train_pac = pac.tracestoInstances(X_train)
    X_test_pac = pac.tracestoInstances(X_test)

    # CHECK FOR FEATURES UNIFORMITY
    features_ = []
    for i in X_train_pac:
        for j in i.keys():
            if j not in features_:
                features_.append(j)
    for i in X_test_pac:
        for j in i.keys():
            if j not in features_:
                features_.append(j)

    X_train_pac_R = []
    for k, element in enumerate(X_train_pac):
        tmp_fea = []
        for key_ in features_:
            tmp_fea.append(element[key_] if key_ in element.keys() else 0)
        X_train_pac_R.append(tmp_fea)
    X_train_pac = X_train_pac_R

    X_test_pac_R = []
    for k, element in enumerate(X_test_pac):
        tmp_fea = []
        for key_ in features_:
            tmp_fea.append(element[key_] if key_ in element.keys() else 0)
        X_test_pac_R.append(tmp_fea)
    X_test_pac = X_test_pac_R

    print("fitting model")
    classifier_pac = pac.fit(y_train, X_train_pac,max_iter)
    del X_train_pac

    print("predicting")
    y_prediction = pac.predict(classifier_pac, X_test_pac)
    del X_test_pac

    return y_test, y_prediction
