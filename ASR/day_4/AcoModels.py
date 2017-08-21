#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
EmptyDistValue = np.float32("inf")


class AcoModel:
    def __init__(self, phone, id):
        self.phone = phone
        self.phone_id = id


class AcoModelSet:
    def __init__(self, phones_file):
        self.name2model = {}
        model_idx = 0
        with open(phones_file, 'r') as fn:
            for line in fn:
                line = line.strip()
                model = AcoModel(line, model_idx)
                self.name2model[model.phone] = model
                model_idx += 1

    def loadPrediction(self, fileName):
        self.predicts = np.load(fileName)

    def loadPosteriors(self, fileName):
        self.posteriors = np.load(fileName)

    def getDist(self, AM, frame):
        predict = self.predicts[frame]
        dist = np.log(predict[AM.phone_id] / self.posteriors[AM.phone_id])
        if dist < 0:
            return -2*dist
        else:
            return -2*dist

    def findModelbyName(self, modelName):
        return self.name2model[modelName]

    def findModelbyId(self, id):
        for key in self.name2model.keys():
            if self.name2model[key] == id:
                return key

    def getUgidCount(self):
        return len(list(self.name2model.keys()))
