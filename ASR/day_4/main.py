#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import GraphMaker
import AcoModels
import FtrFile
import wer
import numpy as np
import time
import os



####################################################################################

class State:
    def __init__(self, model, idx):  # idx is for debug purposes
        self.model = model
        self.word = None
        self.isFinal = False
        self.nextStates = []
        self.idx = idx
        self.bestToken = None

#####################################################################################
# Token
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
        if Tokens[i].alive == True:
            if minDist == None:
                minDist = Tokens[i].dist
                bestToken = Tokens[i]
            elif Tokens[i].dist <= minDist:
                minDist = Tokens[i].dist
                bestToken = Tokens[i]
    return bestToken

######################################################################################

def beamPrunning(nextTokens, thr_common):
    bestToken = findBest(nextTokens)
    bestDist = bestToken.dist
    for i in range(len(nextTokens)):
        if nextTokens[i].alive == True \
                and nextTokens[i].dist >= bestDist + thr_common:
            nextTokens[i].alive = False
    return nextTokens, bestDist

######################################################################################

def statePrunning(nextTokens):
    for i in range(len(nextTokens)):
        state_index = nextTokens[i].state.idx
        if graph[state_index].bestToken == None:
            graph[state_index].bestToken = nextTokens[i]
        else:
            if nextTokens[i].dist <= graph[state_index].bestToken.dist:
                graph[state_index].bestToken.alive = False
                graph[state_index].bestToken = nextTokens[i]
            else:
                nextTokens[i].alive = False
    return nextTokens

#######################################################################################

def virtualNodePass(token, newToken):
    newToken.dist += wd_add
    if newToken.sentence and token.state.word != '<sil>':
        newToken.sentence += '-'
    if token.state.word != '<sil>':
        newToken.sentence += token.state.word
    return newToken

########################################################################################

def decoding(filename, features, AMs, graph, path2predict, thr_common, wd_add):

    print("file_name: " + filename)
    startState = graph[0]
    activeTokens = [Token(startState), ]
    nextTokens = []
    AMs.loadPrediction('{}/{}.npy'.format(path2predict, filename))
    # MAGIC
    for frame in range(features.nSamples):
        ftrCurrentFrameRecord = features.readvec()
        runningBestDist = 0
        for token in activeTokens:
            if token.alive:
                for transitState in token.state.nextStates:
                    newToken = Token(transitState, token.dist, token.sentence)
                    # virtual node pass and penalty for word:
                    if newToken.state.idx == 0:
                        newToken = virtualNodePass(token, newToken)
                        activeTokens.append(newToken)
                        continue
                    # end virtual node pass ...
                    newToken.dist += AMs.getDist(transitState.model, frame)
                    nextTokens.append(newToken)
        #make pruning:
        nextTokens = statePrunning(nextTokens)
        nextTokens, bestDist = beamPrunning(nextTokens, thr_common)

        activeTokens = nextTokens
        for token in nextTokens:
            index = token.state.idx
            if graph[index].bestToken:
                graph[index].bestToken = None
        nextTokens = []

    # finding final Tokens
    finalTokens = []
    for token in activeTokens:
        if token.state.isFinal and token.alive:
            finalTokens.append(token)

    assert len(finalTokens) != 0
    winToken = findBest(finalTokens)
    fn.write(str(filename) + ' ' + winToken.sentence.replace('-', ' ') + '\n')

    return frame

#####################################################################################

def computeWer(testName, resName):

    WER = wer.computeWER(testName, resName)
    print('\n' + '-'*10 + 'RESULT OF RECOGNITION:' + '--'*10)
    print('%WER is {}'.format(WER))

#########################################################
# MAIN...
#########################################################

if __name__ == "__main__":

    lexicon = 'lex.txt'
    phones_dict = 'phones'
    test_mfcc = 'ark,t:txt/test_mfcc.txt'
    path2predict = 'binaryPredict'
    resName = 'decode_results'
    testName = 'test_ref.txt'

    thr_common = 60
    wd_add = 55
    s_time = time.time()

    # 1. Make acoustic model:
    AMs = AcoModels.AcoModelSet(phones_dict)
    AMs.loadPosteriors('posteriors.npy')

    # 2. Load and check graph from lexicon:
    graph = GraphMaker.load_graph(lexicon, AMs)
    GraphMaker.check_graph(graph)

    #3. Decoding:
    print("\nDECODING IS STARTED...")
    num_file = 0
    numbFrame = 0
    with open(resName, 'w') as fn:
        for filename, features in FtrFile.FtrDirectoryReader(test_mfcc):
            if num_file < 1000:
                decoding(filename, features, AMs, graph, path2predict,
                            thr_common, wd_add)
                num_file += 1
                numbFrame += features.nSamples
            else: break

    #4. Compute WER:
    computeWer(testName, resName)

    #5. Print time and RTF:
    time = time.time() - s_time
    minut = int(time / 60)
    second = time - minut * 60
    print("\nTOTAL TIME: {} min {} sec".format(minut, round(second, 2)))
    rtf = time / (numbFrame * 0.01)
    print("RTF: {}".format(round(rtf, 3)))

