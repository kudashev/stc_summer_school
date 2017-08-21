#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import FtrFile
import AcoModels


#########################################
# State, Graph, etc...
#########################################

class State:
    def __init__(self, model, idx):  # idx is for debug purposes
        self.model = model
        self.word = None
        self.isFinal = False
        self.nextStates = []
        self.idx = idx
        self.bestToken = None
        self.currentWord = None


def DictReader(dictName):
    with open(dictName, 'rt') as dict_file:
        for line in dict_file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0]
            phonemes = parts[1:]
            yield key, phonemes


def load_graph(rxfilename, AMs):

    startState = State(None, 0)
    graph = [startState, ]
    stateIdx = 1
    for word, phonemes in DictReader(rxfilename):
        prevState = startState
        for frame in range(len(phonemes)):
            phoneme_model = AMs.findModel(phonemes[frame])
            state = State(phoneme_model, stateIdx)
            state.currentWord = word
            state.nextStates.append(state)  # add loop
            prevState.nextStates.append(state) 
            prevState = state       # prevState is state = true
            graph.append(state)
            stateIdx += 1
        if state:
            state.word = word
            state.isFinal = True
    return graph

#####################################################################################
### Token
#####################################################################################

class Token:
    def __init__(self, state, dist=0.0, sentence=""):
        self.state = state
        self.dist = dist
        self.sentence = sentence
        self.alive = True

##########################################################################################
# Decoder
##########################################################################################

def findBest(Tokens):
    minDist = None
    bestToken = 0
    for i in range(len(Tokens)):
        if Tokens[i].alive:
            if not minDist:
                minDist = Tokens[i].dist
                bestToken = Tokens[i]
            elif Tokens[i].dist <= minDist:
                minDist = Tokens[i].dist
                bestToken = Tokens[i]
    return bestToken

#####################################################################################

def beamPurnning(nextTokens):
    thr_common = 150        #150#150DNstreet#10#150#500#70
    bestToken = findBest(nextTokens)
    bestDist = bestToken.dist
    for i in range(len(nextTokens)):
        if nextTokens[i].alive and nextTokens[i].dist > bestDist + thr_common:
            nextTokens[i].alive = False
    return nextTokens, bestDist

#####################################################################################

def statePrunning(nextTokens):
    for i in range(len(nextTokens)):
        state_index = nextTokens[i].state.idx
        if not graph[state_index].bestToken:
            graph[state_index].bestToken = nextTokens[i]
        else:
            if nextTokens[i].dist <= graph[state_index].bestToken.dist:
                graph[state_index].bestToken.alive = False
                graph[state_index].bestToken = nextTokens[i]
            else:
                nextTokens[i].alive = False
    return nextTokens

###########################################################################################

def recognize(rec_results, features, graph):

    startTime = time.time()
    print("-" * 20)
    startState = graph[0]
    activeTokens = [Token(startState), ]
    nextTokens = []

    for frame in range(features.nSamples):
        ftrCurrentFrameRecord = features.readvec()
        for token in activeTokens:
            if token.alive:
                for transitionState in token.state.nextStates:
                    newToken = Token(transitionState, token.dist, token.sentence)
                    newToken.dist += transitionState.model.dist(ftrCurrentFrameRecord)
                    nextTokens.append(newToken)
        nextTokens = statePrunning(nextTokens)               # Purnning         #
        nextTokens,bestDist = beamPurnning(nextTokens)       #         of       #

        activeTokens = nextTokens

        for token in nextTokens:                           # cleaning the bestTokens
            index = token.state.idx                        #
            if graph[index].bestToken:                     #
                graph[index].bestToken = None              #
        nextTokens = []

    counter = 0
    finalTokens = []
    for token in activeTokens:
        if token.state.isFinal and token.alive:
            finalTokens.append(token)
            counter += 1

    if len(finalTokens) != 0:
        winToken = findBest(finalTokens)
    else:
        winToken = findBest(activeTokens)
        winToken.state.word = winToken.state.currentWord

    print("result: {} {}".format(filename, winToken.state.word))

    endTime = time.time()
    print("time: {} sec".format(round(endTime - startTime, 2)))

    etalons_word = filename.split('_')[0]
    records_word = winToken.state.word.split('_')[0]
    rec_results.append(etalons_word == records_word)

    return frame


###########################################################################
# Main
###########################################################################

if __name__ == "__main__":

    base = 1
    if base == 1:
        file_etalons = "daNetDict.txt"
        records = "ark,t:records_daNet_mfcc.txtftr"
    else:
        file_etalons = "digitDict.txt"
        records = "ark,t:records_digit_mfcc.txtftr"

    file_AMs = "outAM"

    rec_results = []
    s_time = time.time()
    numbFrame = 0

    AMs = AcoModels.loadAcoModelSet(file_AMs)
    graph = load_graph(file_etalons, AMs)
    for filename, features in FtrFile.FtrDirectoryReader(records):
        frame = recognize(rec_results, features, graph)
        numbFrame += frame
    print("-" * 20)
    print("WER is: {}".format(round(1 - sum(rec_results) / len(rec_results), 3)))

    time = time.time() - s_time
    minut = int(time / 60)
    second = time - minut * 60
    print("Total time: {} min {} sec".format(minut, round(second,2)))
    rtf = time / (numbFrame * 0.01)
    print("RTF is: {}".format(round(rtf,3)))
