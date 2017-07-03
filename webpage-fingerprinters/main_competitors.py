# This is part of a Python framework to compliment "Robust Smartphone App Identification Via Encrypted Network Traffic Analysis".
# Copyright (C) 2017  Vincent F. Taylor and Riccardo Spolaor
# See LICENSE for more details.

import pickle

from common_utils import get_results, datasets_files, datasets_ordered, datasets_to_use, read_data_from_pickles

import panchenko_2016
import dyer_2012
import panchenko_2011
import liberatore_2006
import lu_2010
import herrman_2009


folder_of_datasets="."

random_state = 42

down_limit, up_limit = 0, -1

results_file = "results_all"

results = {}


###### CYCLE FOR ALL THE CLASSIFICATIONS#########
for mode_ in ["noisefree","withnoise"]:
    for db_ in datasets_ordered:#datasets_to_use:

        print("loading %s" % datasets_to_use[db_][0])
        X_train_flow, X_train_time, y_train_time = read_data_from_pickles(
            "./%s/%s" % (folder_of_datasets,datasets_files[datasets_to_use[db_][0]][mode_]))
        print("loading %s" % datasets_to_use[db_][1])
        X_test_flow, X_test_time, y_test_time = read_data_from_pickles(
            "./%s/%s" % (folder_of_datasets, datasets_files[datasets_to_use[db_][1]][mode_]))

        instance_of_datasets = "%s__%s_%s_%s" % (db_, mode_, datasets_to_use[db_][0], datasets_to_use[db_][1])

        print ("---------------", instance_of_datasets, "------------------")
	#SAMPLING OF THE DATASET IF NEEDED########
        X_tr_time = X_train_time[down_limit: up_limit]
        X_tr_flow = X_train_flow[down_limit: up_limit]
        y_tr_labels = y_train_time[down_limit: up_limit]

        X_te_time = X_test_time[down_limit: up_limit]
        X_te_flow = X_test_flow[down_limit: up_limit]
        y_te_labels = y_test_time[down_limit: up_limit]

        # I have to use different paramenters for each classifier

        results[instance_of_datasets] = {}

        #####################################################################

        classifier_name = "liberatoreNB2006"
        print("running", classifier_name)
        y_test, y_pred = liberatore_2006.classifier_liberatore_NB(X_tr_flow,
                                                                  y_tr_labels,
                                                                  X_te_flow,
                                                                  y_te_labels,
                                                                  )
        results = get_results(results, instance_of_datasets, classifier_name, y_test, y_pred, results_file)

        #####################################################################

        classifier_name = "liberatoreJaccard2006"
        print("running", classifier_name)
        y_test, y_pred = liberatore_2006.classifier_liberatore_jaccard(X_tr_flow,
                                                                       y_tr_labels,
                                                                       X_te_flow,
                                                                       y_te_labels,
                                                                       )
        results = get_results(results, instance_of_datasets, classifier_name, y_test, y_pred, results_file)

        #####################################################################

        cos_ = [True]#, False]
        norm_ = [True]#, False]
        TF_ = [True]#, False]
        for c in cos_:
            for n in norm_:
                for tf in TF_:
                    classifier_name = "herrman_2009_TF_%s__cos_%s__norm_%s" % (tf, c, n)  ##TESTED
                    print("running", classifier_name)
                    y_test, y_pred = herrman_2009.classifier_herrman2009(X_tr_flow,
                                                                         y_tr_labels,
                                                                         X_te_flow,
                                                                         y_te_labels,
                                                                         cos_=c, TF_=tf, norm=n
                                                                         )
                    results = get_results(results, instance_of_datasets, classifier_name, y_test, y_pred,results_file)



        #####################################################################
        classifier_name = "dyer_2012_time"
        print("running",classifier_name)
        y_test, y_pred = dyer_2012.classifier_dyer2012(X_tr_flow,
                                                       y_tr_labels,
                                                       X_te_flow,
                                                       y_te_labels,
                                                       time_train=X_tr_time,
                                                       time_test=X_te_time
                                                       )
        results = get_results(results, instance_of_datasets, classifier_name, y_test, y_pred,results_file)

        #####################################################################
        classifier_name = "dyer_2012_notime"
        print("running", classifier_name)
        y_test, y_pred = dyer_2012.classifier_dyer2012(X_tr_flow,
                                                       y_tr_labels,
                                                       X_te_flow,
                                                       y_te_labels,
                                                       time_train=None,
                                                       time_test=None
                                                       )
        results = get_results(results, instance_of_datasets, classifier_name, y_test, y_pred,results_file)

        #####################################################################
        classifier_name = "panchenko_2016"
        print("running", classifier_name)
        y_test, y_pred = panchenko_2016.classifier_panchenko2016(X_tr_flow,
                                                                 y_tr_labels,
                                                                 X_te_flow,
                                                                 y_te_labels,
                                                                 )
        results = get_results(results, instance_of_datasets, classifier_name, y_test, y_pred,results_file)

        #####################################################################
        classifier_name = "panchenko_2011"
        print("running", classifier_name)
        y_test, y_pred = panchenko_2011.classifier_panchenko2011(X_tr_flow,
                                                                 y_tr_labels,
                                                                 X_te_flow,
                                                                 y_te_labels,
                                                                 )
        results = get_results(results, instance_of_datasets, classifier_name, y_test, y_pred,results_file)

        #####################################################################
        classifier_name = "lu_2010"  
        print("running", classifier_name)
        y_test, y_pred = lu_2010.classifier_lu2010(X_tr_flow,
                                                   y_tr_labels,
                                                   X_te_flow,
                                                   y_te_labels,
                                                   )
        results = get_results(results, instance_of_datasets, classifier_name, y_test, y_pred,results_file)




