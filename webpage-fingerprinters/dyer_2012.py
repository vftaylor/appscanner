# This is part of a Python framework to compliment "Robust Smartphone App Identification Via Encrypted Network Traffic Analysis".
# Copyright (C) 2017  Vincent F. Taylor and Riccardo Spolaor
# See LICENSE for more details.
########################
#This Python code was freely readapted from the code for "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail" and redistributed under the same license (GNU General Public License).
#Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
#https://github.com/kpdyer/website-fingerprinting/
########################

from sklearn.naive_bayes import BaseNB, GaussianNB, BernoulliNB, MultinomialNB
import numpy as np

from sklearn.preprocessing.data import MinMaxScaler

import sys
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.grid_search import GridSearchCV



class Dyer2012TimeClassifier:
    @staticmethod
    def traceToInstance(trace_):
        maxTime = 0

        # get the duration of the trace
        # TODO

        return [maxTime]


class Dyer2012VNGPlusPlusClassifier:
    @staticmethod
    def roundArbitrary(x, base):
        return int(base * round(float(x) / base))

    @staticmethod
    def traceToInstance(trace_,time_=None):

        # trace ==> array of packets
        instance = {}

        def getPacketDirection(p):
            return "UP" if np.sign(packet_)>0 else "DOWN"

        # Size/Number Markers
        directionCursor = None
        dataCursor = 0
        for packet_ in trace_:  # .getPackets():
            if directionCursor == None:
                directionCursor = getPacketDirection(packet_)  # packet.getDirection()

            if getPacketDirection(packet_) != directionCursor:
                dataKey = 'S' + str(directionCursor) + '-' + str(Dyer2012VNGPlusPlusClassifier.roundArbitrary(dataCursor, 600))
                if not instance.get(dataKey):
                    instance[dataKey] = 0
                instance[dataKey] += 1

                directionCursor = getPacketDirection(packet_)  # packet.getDirection()
                dataCursor = 0

            dataCursor += np.abs(packet_)  # packet.getLength()

        if dataCursor > 0:
            key = 'S' + str(directionCursor) + '-' + str(Dyer2012VNGPlusPlusClassifier.roundArbitrary(dataCursor, 600))
            if not instance.get(key):
                instance[key] = 0
            instance[key] += 1

        def get_bandwidth(trace_, direction="UP"):
            bandwidth = 0
            if direction == "UP":
                for pkt in trace_:
                    bandwidth += np.abs(pkt) if getPacketDirection(pkt) >= "UP" else 0
            elif direction == "DOWN":
                for pkt in trace_:
                    bandwidth += np.abs(pkt) if getPacketDirection(pkt) >= "DOWN" else 0
            return bandwidth

        instance['bandwidthUp'] = get_bandwidth(trace_, direction="UP")
        instance['bandwidthDown'] = get_bandwidth(trace_, direction="DOWN")

        if time_ is not None:
            instance['time'] = time_ #Dyer2012TimeClassifier().traceToInstance(time_)[0]

        #print(instance.keys())
        return instance #[ np.float64(instance[elem]) for elem in instance.keys()]


def dyer2012_tracestoInstances(obj, traces, times=None, fields_given=None):
    instances_1 = []
    fields=[]
    for i, t in enumerate(traces):
        if times is not None:
            instance = obj.traceToInstance(t, times[i])
        else:
            instance = obj.traceToInstance(t)

        for k in instance.keys():
            if k not in fields:
                fields.append(k)

        instances_1.append(instance)
    instances_2 = []

    #print(fields)

    fields = fields if fields_given is None else fields_given

    for i2 in instances_1:
        tmp_=[]
        for f in fields:
            tmp_.append(float(i2[f]) if f in i2.keys() else 0.0)
        instances_2.append(tmp_)

    return instances_2, fields

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def getBestEstimator(self):
        best_score=0
        best_key=None
        for key in self.keys:
            if self.grid_searches[key].best_score_ >= best_score:
                best_score=self.grid_searches[key].best_score_
                best_key=key
        print("best estimator =%s with %s"%(key,best_score))
        return self.grid_searches[key]




    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
            }
            return pd.Series(dict(params.items() + d.items()))

        rows = [row(k, gsc.cv_validation_scores, gsc.parameters)
                for k in self.keys
                for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

def classifier_dyer2012(X_train, y_train, X_test, y_test, time_train=None, time_test=None):

    obj = Dyer2012VNGPlusPlusClassifier()

    X_train, fields = dyer2012_tracestoInstances(obj, X_train, time_train)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    models1 = {
        'Bernoulli': BernoulliNB(),
        'Gaussian': GaussianNB(),
        'Multinomial': MultinomialNB(),
    }

    params1 = {
        'Bernoulli': {},
        'Gaussian': {},
        'Multinomial': {},
        #'SVC': [
        #    {'kernel': ['linear'], 'C': [1, 10]},
         #   {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
        #]
    }

    dyer_NB=MultinomialNB()
    dyer_NB.fit(X_train, y_train)
    del X_train

    #test
    X_test, fields = dyer2012_tracestoInstances(obj, X_test, time_test, fields)
    X_test = scaler.transform(X_test)


    predictions = dyer_NB.predict(X_test)
    del X_test

    labels = []
    for l in y_train:
        if l not in labels:
            labels.append(l)

    return y_test, predictions
