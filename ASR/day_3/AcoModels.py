#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pickle

class AcoModel:
    def dist(self,ftrInput):              #вычисление дистанции
        difference = float(np.sqrt(sum(pow(self.meanValue - ftrInput, 2))))
        return difference

    def __init__(self,sortDir,fnModel):           
        self.name = fnModel.replace('.npy','')        #remove extension
        path = os.path.abspath(sortDir)
        ftrInput = np.load("/"+path+"/"+fnModel)
        self.meanValue = sum(ftrInput)/len(ftrInput)

    pass
pass

class AcoModelSet:

    def findModel(self, modelName):
        return self.name2model[modelName]

    def save(self,fname):
        with open(fname,"wb") as f:
            pickle.dump(self,f)
        pass

    def __init__(self,sortDir,saveToFile):
        self.name2model={}
        print("Training...")
        for fnModel in os.listdir(sortDir):
            model = AcoModel(sortDir, fnModel)
            self.name2model[model.name]=model
        pass

        print("...saving")
        self.save(saveToFile)
        print("Done")
    pass
pass

def loadAcoModelSet(fname):

    with open(fname, "rb") as f:
        q = pickle.load(f)
    return q
