from __future__ import division
from collections import OrderedDict
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

states = [0,[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],1]

def generateTrainingSets(n, s=10):
    """ Generate n training sets of s sequences """
    trainingSets = []
    for _ in range(n):
        trainingSets.append( [ generateTrainingSequence() for _ in range(s) ] )
    return trainingSets


def generateTrainingSequence():
    """ Generate a sequence starting from D with equal probability of going left and right """
    idx = 3
    seq = []
    while idx not in [0,6]:
        seq.append(states[idx])
        if random.random() < 0.5:
            idx += 1
        else:
            idx -= 1
    return (seq, states[idx])


def saveTrainingData():
    """ Save 100 training sets to file """
    trainingSets = generateTrainingSets(100)
    with open('p1_trainingsets.pickle', 'wb') as handle:
        pickle.dump(trainingSets, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadTrainingData():
    """ Load 100 training sets """
    with open('p1_trainingsets.pickle', 'rb') as handle:
        trainingSets = pickle.load(handle)
    return trainingSets


def TD(trainingSet, lambdaValue=1.0, alphaValue=0.1, convergence=True, updatePerSequence=False):
    """ Use TD learning to calculate RMSE between learned value and ideal prediction

    :param trainingSet: a set of 10 observation sequences, each being a list of state vectors and the final outcome
    :param lambdaValue: between 0.0 and 1.0
    :param alphaValue: the learning rate
    :param convergence: continue until convergence or return after one iteration
    :param updatePerSequence: update weights after each sequence or after a whole training set presentation
    :return: final weight vector and its RMSE
    """
    weightVector = np.array([0.5] * 5)
    rmse = 0.0
    prev_rmse = rmse
    while True:
        # Keep presenting training set until weight converges if convergence flag is True
        dw = np.zeros(5)
        for sequence, z in trainingSet:
            # Accumulate delta weights for each sequence
            et = 0
            for t in range(len(sequence)):
                et = et * lambdaValue + np.array(sequence[t])
                Pt = np.dot(weightVector, np.array(sequence[t]))
                Pt1 = np.dot(weightVector, np.array(sequence[t+1])) if t+1 < len(sequence) else z
                dw += (et * alphaValue * (Pt1 - Pt))
            if updatePerSequence:
                weightVector += dw
                dw = np.zeros(5)
        if not updatePerSequence:
            weightVector += (dw/10.0)
        prev_rmse = rmse
        ideal = np.linspace(0, 1, 7)[1:-1]
        rmse = (sum((weightVector - ideal)**2.0)/5.0)**0.5
        if (abs(prev_rmse - rmse) < 0.001) or not convergence:
            break
    return weightVector, rmse


def experiment1(trainingSet, lambdas, plot=True):
    """ Experiment 1
    Accumulate weight vectors over sequences and use to update weight only after the complete presentation of the set.
    Present the set repeatedly until the weight vector converges. For small alpha, the weight vector always converges
    to the same final value.

    Ideal predictions for non-terminal states can be computed as described in 4.1 [1/6, 1/3, 1/2, 2/3, 5/6].
    np.linspace(0, 1, n_states)[1:-1], n_state = 7
    """
    errors = []
    for l in lambdas:
        averageError = 0
        for trainingSet in trainingSets:
            weightVector, error = TD(trainingSet, lambdaValue=l)
            averageError += error
        averageError /= 100.0
        print(l, averageError)
        errors.append((l, averageError))

    if plot:
        d = OrderedDict(errors)
        plt.plot(d.keys(), d.values(), marker='o')
        plt.xlabel(u'\u03BB')
        plt.ylabel('ERROR')
        plt.xticks(lambdas)
        plt.show()

    return errors


def errorsByAlpha(trainingSets, lambdas, plot=True):
    """ Experiment 2
    For several lambdas test varying learning rates. Update weights after each sequence without converging.
    """
    errors = {}
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    for l in lambdas:
        errors[l] = []
        for alpha in alphas:
            averageError = 0
            for trainingSet in trainingSets:
                weightVector, error = TD(trainingSet, lambdaValue=l, alphaValue=alpha, convergence=False, updatePerSequence=True)
                averageError += error
            averageError /= 100.0
            print("lambda {}, alpha {}, error {}".format(l, alpha, averageError))
            errors[l].append((alpha, averageError))

    if plot:
        for key in lambdas:
            d = OrderedDict(errors[key])
            plt.plot(d.keys(), d.values(), marker='o')
        plt.xlabel(u'\u03B1')
        plt.ylabel('ERROR')
        plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6])
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        plt.ylim((0, 0.7))
        plt.legend([u'\u03BB = 0', u'\u03BB = .3', u'\u03BB = .8', u'\u03BB = 1'], loc='upper left')
        plt.show()

    return errors


def experiment2(trainingSets, lambdas, plot=True):
    """ Experiment 2
    Use the results from errorsByAlpha to get the best alphas for various lambdas. Get the average error across all
    training sets for various lambdas with their best alphas.
    """
    exp2Results = errorsByAlpha(trainingSets, lambdas=lambdas, plot=False)
    bestAlphas = {}
    for l, errors in exp2Results.iteritems():
        d = dict(errors)
        bestAlphas[l] = min(d, key=d.get)

    errors = []
    for l in lambdas:
        averageError = 0
        for trainingSet in trainingSets:
            weightVector, error = TD(trainingSet, lambdaValue=l, alphaValue=bestAlphas.get(l), convergence=False, updatePerSequence=True)
            averageError += error
        averageError /= 100.0
        print(l, averageError)
        errors.append((l, averageError))

    if plot:
        d = OrderedDict(errors)
        plt.plot(d.keys(), d.values(), marker='o')
        plt.xlabel(u'\u03BB')
        plt.ylabel(u'ERROR USING BEST \u03B1')
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.show()

    return errors

if __name__ == "__main__":
    # saveTrainingData()
    trainingSets = loadTrainingData()
    experiment1(trainingSets, lambdas=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], plot=True)
    errorsByAlpha(trainingSets, lambdas=[0.0, 0.3, 0.8, 1.0], plot=True)
    experiment2(trainingSets, lambdas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], plot=True)


