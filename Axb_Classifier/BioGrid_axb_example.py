#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import axb_classifiers as axb
from axb_utils import axb_score_prediction
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import KFold

path_start = './biogrid' #replace with path string to folder containing BioGrid dataset
organisms = ['AT', 'CE', 'DM', 'EC', 'HS', 'MM', 'RN', 'SC', 'SP']
models = ['ER', 'ERDD', 'GEO', 'GEOGD', 'HGG', 'SF', 'SFDD', 'Sticky']

data = []
features = []
labels = []

for o, organism in enumerate(organisms):
    for m, model in enumerate(models):
        with open(f'{path_start}{organism}{model}', 'r') as file:
            for line in file:
                example = []
                for value in line.split():
                    example.append(float(value))
                features.append(example)
                labels.append(o * 8 + m) #enumerate classes

mas = MaxAbsScaler()
mas.fit(features)
features = mas.transform(features)
kf = KFold(n_splits=10, shuffle=True, random_state=1)

print('Accuracy Subscores:')

for i_train, i_test in kf.split(features):
    x_train, y_train = np.array([features[i] for i in i_train]), np.array([labels[i] for i in i_train])
    x_test, y_test = np.array([features[i] for i in i_test]), np.array([labels[i] for i in i_test])

    classifier = axb.axb_classifier(split='density', ri=.01, dr=.005, rm=.5)
    classifier.fit(x_train, y_train, 'hsm')
    score = axb_score_prediction(classifier, x_test, y_test)
    print(score)
    data.append(score)

print('Final Acccuracy:', sum(data) / len(data))
