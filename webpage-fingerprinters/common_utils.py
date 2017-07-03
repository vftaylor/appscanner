# This is part of a Python framework to compliment "Robust Smartphone App Identification Via Encrypted Network Traffic Analysis".
# Copyright (C) 2017  Vincent F. Taylor and Riccardo Spolaor
# See LICENSE for more details.


import cPickle, gzip,pickle

from sklearn.metrics.classification import accuracy_score, precision_score, recall_score, f1_score


##DIRECTORY OF RESULTS
dir_results="."

def get_results(results, instance_of_datasets, classifier_name, y_true, y_pred, file_dump):
    tmp_ = {"y_pred": y_pred,
                                                      "y_true": y_true,
                                                      "accuracy": accuracy_score(y_true, y_pred),
                                                      "precision_micro": precision_score(y_true, y_pred,
                                                                                         average="micro"),
                                                      "precision_macro": precision_score(y_true, y_pred,
                                                                                         average="macro"),
                                                      "recall_micro": recall_score(y_true, y_pred, average="micro"),
                                                      "recall_macro": recall_score(y_true, y_pred, average="macro"),
                                                      "f1_micro": f1_score(y_true, y_pred, average="micro"),
                                                      "f1_macro": f1_score(y_true, y_pred, average="macro")
                                                      }

    cPickle.dump(tmp_, gzip.open("%s/single_%s_%s_%s.zcp"%(dir_results,file_dump,instance_of_datasets, classifier_name), "wb+"))
    results[instance_of_datasets][classifier_name]=tmp_
    print(classifier_name,
          "accuracy", results[instance_of_datasets][classifier_name]["accuracy"],
          "f1 score_micro", results[instance_of_datasets][classifier_name]["f1_micro"],
          "precision_micro", results[instance_of_datasets][classifier_name]["precision_micro"],
          "recall_micro", results[instance_of_datasets][classifier_name]["recall_micro"],
          "f1 score_macro", results[instance_of_datasets][classifier_name]["f1_macro"],
          "precision_macro", results[instance_of_datasets][classifier_name]["precision_macro"],
          "recall_macro", results[instance_of_datasets][classifier_name]["recall_macro"]
          )
    cPickle.dump(results, gzip.open(dir_results+"/"+file_dump, "wb+"))
    return results


def read_data_from_pickles(path_):
    data_ = pickle.load(open(path_))
    X_flows, X_time, labels = [], [], []
    for label, time, flow in data_:
        X_flows.append(flow)
        X_time.append(time[2])  # there are 2 start time, and duration
        labels.append(label)
    return X_flows, X_time, labels


########### RENAME THE FILES IF NEEDED, 
########### IT WORKS WITH FLOW DATASET, IT DOESN'T WORK WITH STATISTICAL DATASETS
######################### 
datasets_files = {
    "lg-new-65": {  # Dataset-3
        "noisefree": "lg-new-65-noisefree-timing.p",
        "withnoise": "lg-new-65-withnoise-timing.p",
    },
    "lg-new-new-65": {  # Dataset-5a
        "noisefree": "lg-new-new-65-noisefree-timing.p",
        "withnoise": "lg-new-new-65-withnoise-timing.p",
    },
    "lg-new-new-110": {  # Dataset-5
        "noisefree": "lg-new-new-110-noisefree-timing.p",
        "withnoise": "lg-new-new-110-withnoise-timing.p",
    },
    "motog-new-65": {  # Dataset-2
        "noisefree": "motog-new-65-noisefree-timing.p",
        "withnoise": "motog-new-65-withnoise-timing.p",
    },
    "motog-new-new-65": {  # Dataset-4a
        "noisefree": "motog-new-new-65-noisefree-timing.p",
        "withnoise": "motog-new-new-65-withnoise-timing.p",
    },
    "motog-new-new-110": {  # Dataset-4
        "noisefree": "motog-new-new-110-noisefree-timing.p",
        "withnoise": "motog-new-new-110-withnoise-timing.p",
    },
    "motog-old-65": {  # Dataset-1a
        "noisefree": "motog-old-65-noisefree-timing.p",
        "withnoise": "motog-old-65-withnoise-timing.p",
    },
    "motog-old-110": {  # Dataset-5
        "noisefree": "motog-old-110-noisefree-timing.p",
        "withnoise": "motog-old-110-withnoise-timing.p",
    },
}

#####REPLICATION OF THE EXPERIMENTS IN THE PAPER#################
datasets_to_use = {"TIME": (
    "motog-old-65",  # Dataset-1a
    "motog-new-65"  # Dataset-2
),
    "D-110": (
        "motog-new-new-110",  # Dataset-4
        "lg-new-new-110"  # Dataset-5
    ),
    "D-110A": ("motog-new-new-65", "lg-new-new-65"),  # Dataset-4a Dataset-5a 
    "D-65": ("motog-new-65", "lg-new-65"),  # Dataset-2 Dataset-3 
    "V-LG": ("lg-new-65", "lg-new-new-65"),  # Dataset-3 Dataset-5a 
    "V-MG": ("motog-new-65", "motog-new-new-65"),  # Dataset-2 Dataset-4a 
    "DV-110":("motog-old-110","lg-new-new-110"), #Dataset-1 Dataset-5 
    "DV-65": ("motog-old-65", "lg-new-new-65"),  # Dataset-1a Dataset-5a
}

datasets_ordered=["TIME",
    "DV-110",  #Dataset-1 Dataset-5
    "D-110",
    "D-110A",#: ("motog-new-new-65", "lg-new-new-65"),  # Dataset-4a Dataset-5a 38.4 35.2 35.1 37.7 Device 65
    "D-65",#: ("motog-new-65", "lg-new-65"),  # Dataset-2 Dataset-3 43.5 38.3 39.0 39.6 Device 65
    "V-LG",#: ("lg-new-65", "lg-new-new-65"),  # Dataset-3 Dataset-5a 33.0 31.0 30.0 30.3 App versions 65
    "V-MG",#: ("motog-new-65", "motog-new-new-65"),  # Dataset-2 Dataset-4a 34.8 32.1 32.1 32.7 App versions 65
    "DV-65"
 ]

