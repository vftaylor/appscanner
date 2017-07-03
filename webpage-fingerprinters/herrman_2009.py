# This is part of a Python framework to compliment "Robust Smartphone App Identification Via Encrypted Network Traffic Analysis".
# Copyright (C) 2017  Vincent F. Taylor and Riccardo Spolaor
# See LICENSE for more details.
########################

from __future__ import print_function
import numpy as np
from numpy import math
from sklearn.naive_bayes import MultinomialNB


def herrmann_data(X_, TF=True, cos=False, normalization=False):
    X = []
    for x in X_:
        data = {"%s" % x: 0 for x in range(1, 1515)}
        for k in x:
            data["%s" % np.abs(k)] += 1
        tmp_ = []
        for x in range(1, 1515):
            tmp_.append(data["%s" % x])
        X.append(tmp_)

    if normalization == True:
        X_t = []
        for x in X:
            x_T = []
            for x_ in x:
                x_tmp = (x_ * 1.0) / sum(x)
                x_T.append(x_tmp)
            X_t.append(x_T)
        X = X_t

    if TF is True:
        X_t = []
        for x in X:
            x_tmp_ = []
            for k in x:
                x_tmp_.append(math.log(1.0 + k, math.e))
            X_t.append(x_tmp_)
        X = X_t
        return X

    if cos is True:
        instances = X
        big_X = []
        for k, instance in enumerate(instances):
            X_T = []
            for x in instance:
                # Apply TF Transformation
                t_tmp = math.log(1.0 + x, 2)
                X_T.append(t_tmp)

            # Store Euclidean Length for Cosine Normalisation (Section 4.5.2)
            euclideanLength = 0
            for attribute in X_T:
                euclideanLength += 1.0 * attribute * attribute
            euclideanLength = math.sqrt(euclideanLength)

            X_T2 = []
            for attribute in X_T:
                # Apply Cosine Normalisation
                t_tmp = 1.0 * attribute / euclideanLength
                X_T2.append(t_tmp)
            big_X.append(X_T2)
        X = big_X

    return X


def classifier_herrman2009(X_train, y_train, X_test, y_test, cos_=True, TF_=True, norm=True):
    X_train_h = herrmann_data(X_train, TF=TF_, cos=cos_, normalization=norm)

    herman_MNB = MultinomialNB()
    herman_MNB.fit(X_train_h, y_train)
    del X_train_h

    X_test_h = herrmann_data(X_test, TF=TF_, cos=cos_, normalization=norm)
    predictions = herman_MNB.predict(X_test_h)
    del X_test_h

    labels = []
    for l in y_train:
        if l not in labels:
            labels.append(l)

    return y_test, predictions
