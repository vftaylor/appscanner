import cPickle as pickle
from sklearn import svm, ensemble
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np

##########
##########
    
TRAINING_PICKLE = 'motog-old-65-withnoise-statistical.p'      # 1a
TESTING_PICKLE  = 'motog-new-65-withnoise-statistical.p'      # 2

print 'Loading pickles...'
trainingflowlist = pickle.load(open(TRAINING_PICKLE, 'rb'))
testingflowlist = pickle.load(open(TESTING_PICKLE, 'rb'))
print 'Done...'
print ''

print 'Training with ' + TRAINING_PICKLE + ': ' + str(len(trainingflowlist))
print 'Testing with ' + TESTING_PICKLE + ': ' + str(len(testingflowlist))
print ''

for THR in range(10):

    p = []
    r = []
    f = []
    a = []
    c = []

    for i in range(10):
        print i
        ########## PREPARE STUFF
        trainingexamples = []
        classifier = ensemble.RandomForestClassifier()
        classifier2 = ensemble.RandomForestClassifier()


        ########## GET FLOWS
        for package, time, flow in trainingflowlist:
            trainingexamples.append((flow, package))
        #print ''


        ########## SHUFFLE DATA to ensure classes are "evenly" distributed
        random.shuffle(trainingexamples)


        ########## TRAINING PART 1
        X1_train = []
        y1_train = []

        for flow, package in trainingexamples[:int(float(len(trainingexamples))/2)]:       
            X1_train.append(flow)
            y1_train.append(package)

        #print 'Fitting classifier...'
        classifier.fit(X1_train, y1_train)
        #print 'Classifier fitted!'
        #print ''


        ########## TRAINING PART 2 (REINFORCEMENT)
        X2_train = []
        y2_train = []

        count1 = 0
        count2 = 0

        for flow, package in trainingexamples[int(float(len(trainingexamples))/2):]:
            prediction = classifier.predict(flow)

            X2_train.append(flow)
            if (prediction == package):
                y2_train.append(package)
                count1 += 1
            else:
                y2_train.append('ambiguous')
                count2 += 1

        #print count1
        #print count2

        #print 'Fitting 2nd classifier...'
        classifier2.fit(X2_train, y2_train)
        #print '2nd classifier fitted!'
        #print ''

                
        ########## TESTING

        threshold = float(THR)/10

        X_test = []
        y_test = []

        totalflows = 0
        consideredflows = 0
            
        for package, time, flow in testingflowlist:
            prediction = classifier2.predict(flow)

            if (prediction != 'ambiguous'):
                prediction_probability = max(classifier2.predict_proba(flow)[0])
                totalflows += 1
                
                if (prediction_probability >= threshold):
                    consideredflows += 1

                    X_test.append(flow)
                    y_test.append(package)

        y_pred = classifier2.predict(X_test)

        p.append(precision_score(y_test, y_pred, average="macro")*100)
        r.append(recall_score(y_test, y_pred, average="macro")*100)
        f.append(f1_score(y_test, y_pred, average="macro")*100)
        a.append(accuracy_score(y_test, y_pred)*100)
        c.append(float(consideredflows)*100/totalflows)

    print 'Threshold: ' + str(threshold)
    print np.mean(p)
    print np.mean(r)
    print np.mean(f)
    print np.mean(a)
    print np.mean(c)
    print ''
