#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import FtrFile

#########################################
# State, Graph, etc...
#########################################

class State:
    def __init__(self, ftr, idx):  # idx is for debug purposes
        self.ftr = ftr
        self.word = None
        self.isFinal = False
        self.nextStates = []
        self.idx = idx


def load_graph(rxfilename):
    startState = State(None, 0)
    graph = [startState, ]
    stateIdx = 1
    for word, features in FtrFile.FtrDirectoryReader(rxfilename):
        prevState = startState
        for frame in range(features.nSamples):
            state = State(features.readvec(), stateIdx)
            state.nextStates.append(state)  # add loop
            prevState.nextStates.append(state)
            prevState = state  # prevState is state = true
            graph.append(state)
            stateIdx += 1
        if state:
            state.word = word
            state.isFinal = True
    return graph

###########################################################################
# Token
###########################################################################

class Token:
    def __init__(self, state, dist=0.0, sentence=""):
        self.state = state
        self.dist = dist
        self.sentence = sentence


###########################################################################
# Decoder
###########################################################################

def distance(X, Y):
    result = float(np.sqrt(pow(X - Y, 2)))
    return result


def recognize(filename, features, graph):
    print("Recognizing file '{}', samples={}".format(filename, features.nSamples))

    startState = graph[0]
    activeTokens = [Token(startState), ]
    nextTokens = []

    for frame in range(features.nSamples):
        ftrCurrentFrameRecord = features.readvec()
        for token in activeTokens:
            for transitionState in token.state.nextStates:
                newToken = Token(transitionState, token.dist, token.sentence)
                newToken.dist += distance(ftrCurrentFrameRecord, transitionState.ftr)
                nextTokens.append(newToken)
        activeTokens = nextTokens
        nextTokens = []

    finalTokens = []
    for token in activeTokens:
        if token.state.isFinal:
            finalTokens.append(token)
    minDist = finalTokens[0].dist
    for token in finalTokens:
        if token.dist <= minDist:
            winToken = token
            minDist = token.dist
    print('\n' + '-'*50)
    print("WIN TOKEN: state.word = '{}',"
          " dist = {}, ".format(winToken.state.word, round(winToken.dist, 3)))
    print('\n' + '-' * 50)

###########################################################################
# Main
###########################################################################

if __name__ == "__main__":
    etalons = "ark,t:etalons_mfcc.txtftr"
    records = "ark,t:record_mfcc.txtftr"

    graph = load_graph(etalons)

    for filename, features in FtrFile.FtrDirectoryReader(records):
        recognize(filename, features, graph)


