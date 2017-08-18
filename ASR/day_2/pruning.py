#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import FtrFile


###################################################################################
# State, Graph, etc...
###################################################################################

class State:
    def __init__(self, ftr, idx):  # idx is for debug purposes
        self.ftr = ftr
        self.word = None
        self.isFinal = False
        self.nextStates = []
        self.idx = idx
        self.bestToken = None
        self.currentWord = None


"""
        self.ftr = ftr           # вектор признаков узла 
        self.isFinal = False     # является ли этот узел финальнвм в слове
        self.word = None         # слово эталона (назначается только для финального узла)       
        self.nextStates = []     # список следующих узлов
        self.idx = idx           # индекс узла
        self.bestToken = None    # лучший токен (по минимуму дистанции) в узле
        self.currentWord = None  # текущее слово эталона
"""

def load_graph(rxfilename):
    startState = State(None, 0)
    graph = [startState, ]
    stateIdx = 1
    for word, features in FtrFile.FtrDirectoryReader(rxfilename):
        prevState = startState
        flag = 0
        for frame in range(features.nSamples):
            state = State(features.readvec(), stateIdx)
            state.currentWord = word
            state.nextStates.append(state)              # add loop
            prevState.nextStates.append(state)
            prevState = state
            if flag >= 2:                                     # add the
                graph[stateIdx-2].nextStates.append(state)    # additional transition
            if flag >= 3:
                graph[stateIdx-3].nextStates.append(state)
            graph.append(state)
            stateIdx += 1
            flag += 1
        if state:
            state.word = word
            state.isFinal = True
    return graph


def check_graph(graph):
    assert len(graph) > 0, "graph is empty."
    assert graph[0].ftr is None \
        and graph[0].word is None \
        and not graph[0].isFinal, "broken start state in graph."
    idx = 0
    for state in graph:
        assert state.idx == idx
        idx += 1
        assert (state.isFinal and state.word is not None) \
            or (not state.isFinal and state.word is None)


def print_graph(graph):
    print("*** DEBUG. GRAPH ***")
    with open('graph.txt', 'w') as fn:
        np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
        for state in graph:
            nextStatesIdxs = [s.idx for s in state.nextStates]
            fn.write("State: idx={} word={} isFinal={} nextStatesIdxs={} ftr={} \n".format(
                state.idx, state.word, state.isFinal, nextStatesIdxs, state.ftr))
    print("*** END DEBUG. GRAPH ***")

#################################################################################
# Token
#################################################################################

class Token:
    def __init__(self, state, dist=0.0, sentence=""):
        self.state = state
        self.dist = dist
        self.sentence = sentence
        self.alive = True

##################################################################################
# Decoder
##################################################################################

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
    thr_common = 70
    bestToken = findBest(nextTokens)
    bestDist = bestToken.dist
    for i in range(len(nextTokens)):
        if nextTokens[i].alive and nextTokens[i].dist > bestDist + thr_common :
            nextTokens[i].alive = False
    return nextTokens

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

#####################################################################################

def distance(X, Y):
    result = float(np.sqrt(sum(pow(X - Y, 2))))
    return result

#####################################################################################

def recognize(features, graph, rec_results):

    print("-" * 20)
    startTime = time.time()
    startState = graph[0]
    activeTokens = [Token(startState), ]
    nextTokens = []

    for frame in range(features.nSamples):
        ftrCurrentFrameRecord = features.readvec()
        for token in activeTokens:
            if token.alive:
                for transitionState in token.state.nextStates:
                    newToken = Token(transitionState, token.dist, token.sentence)
                    newToken.dist += distance(ftrCurrentFrameRecord, transitionState.ftr)
                    nextTokens.append(newToken)
        nextTokens = statePrunning(nextTokens)               # Purnning         #
        nextTokens = beamPurnning(nextTokens)       #         of       #

        activeTokens = nextTokens

        for token in nextTokens:                           # cleaning the bestTokens
            index = token.state.idx                        #
            if graph[index].bestToken:                     #
                graph[index].bestToken = None              #
        nextTokens = []

    finalTokens = []
    for token in activeTokens:
        if token.state.isFinal and token.alive:
            finalTokens.append(token)

    if len(finalTokens) != 0:
        winToken = findBest(finalTokens)
    else:
        winToken = findBest(activeTokens)
        winToken.state.word = winToken.state.currentWord

    print("result: {} {}".format(filename, winToken.state.word))

    endTime = time.time()
    print("time: {} sec".format(round(endTime-startTime, 2)))

    etalons_word = filename.split('_')[0]
    records_word = winToken.state.word.split('_')[0]
    rec_results.append(etalons_word == records_word)

    return frame


########################################################################
# Main
########################################################################

if __name__ == "__main__":
    etalons = "ark,t:etalons_mfcc.txtftr"
    records = "ark,t:records_mfcc.txtftr"

    rec_results = []

    s_time = time.time()
    numbFrame = 0

    graph = load_graph(etalons)

    check_graph(graph)
    print_graph(graph)

    for filename, features in FtrFile.FtrDirectoryReader(records):
        frame = recognize(features, graph, rec_results)
        numbFrame += frame

    print("WER is: {}".format(sum(rec_results)/len(rec_results)))
    e_time = time.time()
    time = e_time-s_time
    minut = int(time/60)
    second = time-minut*60
    print("Total time: {} min {} sec".format(minut, second))
    rtf = time/(numbFrame*0.01)
    print("RTF is: {}".format(rtf))
