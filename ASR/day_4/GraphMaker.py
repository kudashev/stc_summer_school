# -*- coding: utf-8 -*-

import numpy as np

class State:
    def __init__(self, model, idx):  # idx is for debug purposes
        self.model = model
        self.word = None
        self.isFinal = False
        self.nextStates = []
        self.idx = idx
        self.bestToken = None
        self.currentWord = None

####################################################################################

def DictReader(dictName):
    with open(dictName, 'rt') as dict_file:
        for line in dict_file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0]
            phonemes = parts[1:]
            yield word, phonemes

####################################################################################

def load_graph(rxfilename, AMs):
    startState = State(None, 0)
    graph = [startState, ]
    stateIdx = 1
    for word, phones in DictReader(rxfilename):
        prevState = startState
        for frame in range(len(phones)):
            phoneme_model = AMs.findModelbyName(phones[frame])
            state = State(phoneme_model, stateIdx)
            state.currentWord = word
            state.nextStates.append(state)  # add loop
            prevState.nextStates.append(state)
            prevState = state  # prevState is state = true
            graph.append(state)
            stateIdx += 1
        if state:
            state.word = word
            state.isFinal = True
            state.nextStates.append(startState)
    return graph

#####################################################################################

def check_graph(graph):
    assert len(graph) > 0, "graph is empty."
    assert graph[0].model is None \
           and graph[0].word is None \
           and not graph[0].isFinal, "broken start state in graph."
    idx = 0
    for state in graph:
        assert state.idx == idx
        idx += 1
        assert (state.isFinal and state.word is not None) \
               or (not state.isFinal and state.word is None)

