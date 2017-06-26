import cPickle as pickle
from sklearn import svm, ensemble
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np

##########
##########
    
TRAINING_PICKLE = 'motog-old-110-noisefree-statistical.p'     # 1
TESTING_PICKLE  = 'lg-new-new-110-noisefree-statistical.p'    # 5

print 'Loading pickles...'
trainingflowlist = pickle.load(open(TRAINING_PICKLE, 'rb'))
testingflowlist = pickle.load(open(TESTING_PICKLE, 'rb'))
print 'Done...'
print ''

print 'Training with ' + TRAINING_PICKLE + ': ' + str(len(trainingflowlist))
print 'Testing with ' + TESTING_PICKLE + ': ' + str(len(testingflowlist))
print ''

p = []
r = []
f = []
a = []

for i in range(50):
    ########## PREPARE STUFF
    trainingexamples = []
    #classifier = svm.SVC(gamma=0.001, C=100, probability=True)
    classifier = ensemble.RandomForestClassifier()


    ########## GET FLOWS
    for package, time, flow in trainingflowlist:
        trainingexamples.append((flow, package))


    ########## SHUFFLE DATA to ensure classes are "evenly" distributed
    random.shuffle(trainingexamples)


    ########## TRAINING
    X_train = []
    y_train = []

    for flow, package in trainingexamples:       
        X_train.append(flow)
        y_train.append(package)

    print 'Fitting classifier...'
    classifier.fit(X_train, y_train)
    print 'Classifier fitted!'
    print ''

            
    ########## TESTING

    X_test = []
    y_test = []

    for package, time, flow in testingflowlist:
        X_test.append(flow)
        y_test.append(package)

    y_pred = classifier.predict(X_test)

    print(precision_score(y_test, y_pred, average="macro"))
    print(recall_score(y_test, y_pred, average="macro"))
    print(f1_score(y_test, y_pred, average="macro"))
    print(accuracy_score(y_test, y_pred))
    print ''

    p.append(precision_score(y_test, y_pred, average="macro"))
    r.append(recall_score(y_test, y_pred, average="macro"))
    f.append(f1_score(y_test, y_pred, average="macro"))
    a.append(accuracy_score(y_test, y_pred))


print p
print r
print f
print a
print ''

print np.mean(p)
print np.mean(r)
print np.mean(f)
print np.mean(a)
