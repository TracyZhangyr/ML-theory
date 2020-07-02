#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import axb_classifiers as axb

def axb_score_prediction(classifier, examples, labels):
    """Calculates accuracy of trained axb classifier given labelled samples

    Parameters
    ----------
    classifier : axb_classifier
        Trained axb model used to predict labels of given examples

    examples : np.ndarray (samples x features)
        Features given to the classifier to make predictions

    labels : np.ndarray (1-dimensional)
        True class assigned to each example for calculating accuracy (maybe add confusion matrix later on)

    """
    confidences = classifier.new_sample_prediction_confidence(examples.transpose())
    predictions = [np.argmax(l) for l in confidences]
    correct = 0
    right = {}
    total = {}

    for real, pred in zip(labels, predictions):
        correct += pred == real
        if real in total:
            total[real] += 1
            right[real] += pred == real
        else:
            total[real] = 1
            right[real] = int(pred == real)

    return correct / len(predictions)
