# This is part of a Python framework to compliment "Robust Smartphone App Identification Via Encrypted Network Traffic Analysis".
# Copyright (C) 2017  Vincent F. Taylor and Riccardo Spolaor
# See LICENSE for more details.
########################
#this code implements the webpage fingerprinting in Panchenko, A.; Lanze, F.; Zinnen, A.; Henze, M.; Pennekamp, J.; Wehrle, K.; Engel, T.: Website Fingerprinting at Internet Scale In Proceedings of the 23rd Internet Society (ISOC) Network and Distributed System Security Symposium (NDSS 2016), San Diego, USA, February 2016. ISBN: 1-891562-41-X
#the code about has been readapted from A. Panchenko homepage (http://lorre.uni.lu/~andriy/zwiebelfreunde/)
########################

import itertools
import numpy as np
import math



# Panchenko CUMUL n=100
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.svm.classes import LinearSVC, SVC


def outlier_removal(set_labels, data_, class_labels):
    # outlier removal
    merged_dataset = []
    for i, elem in enumerate(data_):
        incomingsize = np.sum([np.abs(pkt) if np.sign(pkt) < 0 else 0 for pkt in elem])
        merged_dataset.append((incomingsize, set_labels[i], data_[i], class_labels[i]))

    instances = sorted(merged_dataset, key=lambda (v, x, y, z): v)

    # remove based on quantiles


    # calculate quantiles
    q2l = ((len(instances) + 1) / 2)
    q1l = int(math.ceil((q2l / 2)))
    q3l = int(math.ceil((3 * (q2l / 2)))) - 1
    if q1l < 0:
        q1l = 0
    if q3l < 0:
        q3l = 0
    if q3l >= len(instances):
        q3l = len(instances) - 1

    q1 = instances[q1l][0]
    q3 = instances[q3l][0]

    remove = []
    # remove outlier based on quantile metric

    y_train = []
    X_train = []
    y_test = []
    X_test = []
    for incomingsize, set_label, instance, class_label in instances:
        if (q1 - 1.5 * (q3 - q1)) < incomingsize < (q3 + 1.5 * (q3 - q1)):
            if set_label == "train":
                y_train.append(class_label)
                X_train.append(instance)
            elif set_label == "test":
                y_test.append(class_label)
                X_test.append(instance)

    return y_train, X_train, y_test, X_test


def features_extraction(labels, data, separateClassifier=True, featuresCount=100):
    # instance type is a list of packets' sizes

    features_ = []
    labels_=[]
    for label, instance in zip(labels, data):
        instance_features=[]

        total = []
        cum = []
        pos = []
        neg = []
        inSize = 0
        outSize = 0
        inCount = 0
        outCount = 0

        # Process trace
        for packet in itertools.islice(instance, None):
            packetsize = int(packet)

            # incoming packets
            if packetsize > 0:
                inSize += packetsize
                inCount += 1
                # cumulated packetsizes
                if len(cum) == 0:
                    cum.append(packetsize)
                    total.append(packetsize)
                    pos.append(packetsize)
                    neg.append(0)
                else:
                    cum.append(cum[-1] + packetsize)
                    total.append(total[-1] + abs(packetsize))
                    pos.append(pos[-1] + packetsize)
                    neg.append(neg[-1] + 0)

            # outgoing packets
            if packetsize < 0:
                outSize += abs(packetsize)
                outCount += 1
                if len(cum) == 0:
                    cum.append(packetsize)
                    total.append(abs(packetsize))
                    pos.append(0)
                    neg.append(abs(packetsize))
                else:
                    cum.append(cum[-1] + packetsize)
                    total.append(total[-1] + abs(packetsize))
                    pos.append(pos[-1] + 0)
                    neg.append(neg[-1] + abs(packetsize))

        # add feature
        instance_features.append(inCount)
        instance_features.append(outCount)
        instance_features.append(outSize)
        instance_features.append(inSize)

        if separateClassifier is True:
            # cumulative in and out
            posFeatures = np.interp(np.linspace(total[0], total[-1], featuresCount / 2), total, pos)
            negFeatures = np.interp(np.linspace(total[0], total[-1], featuresCount / 2), total, neg)
            for el in itertools.islice(posFeatures, None):
                instance_features.append(el)
            for el in itertools.islice(negFeatures, None):
                instance_features.append(el)
        else:
            # cumulative in one
            cumFeatures = np.interp(np.linspace(total[0], total[-1], featuresCount + 1), total, cum)
            for el in itertools.islice(cumFeatures, 1, None):
                instance_features.append(el)

        features_.append(instance_features)
        labels_.append(label)

    return labels, features_


def classifier_panchenko2016(X_train, y_train, X_test, y_test, separateClassifier=False):
    train_or_test_labels = ["train" for i in y_train] + ["test" for i in y_test]
    y_train, X_train, y_test, X_test = outlier_removal(train_or_test_labels, X_train + X_test, y_train + y_test)

    y_train, X_train = features_extraction(y_train, X_train, separateClassifier=separateClassifier, featuresCount=100)

    y_test, X_test = features_extraction(y_test, X_test, separateClassifier=separateClassifier, featuresCount=100)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = SVC(kernel="rbf", C=2e11, gamma=2e-1, max_iter=5000, class_weight="balanced", verbose=1)

    print("fitting")
    classifier.fit(X_train, y_train)

    print("testing")
    y_predictions = classifier.predict(X_test) #, y_test)

    return y_test, y_predictions
