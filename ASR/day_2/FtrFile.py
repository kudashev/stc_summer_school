# -*- coding: utf-8 -*-

import numpy as np


def FtrDirectoryReader(rxfilename):
    if not rxfilename.startswith('ark,t:'):
        raise ValueError("failed to parse rxfilename, "
                         "only 'ark,t' type is supported")
    type, path = rxfilename.split(':', 1)
    for fname, reader in ArkReader(path):
        yield fname, reader


def ArkReader(path):
    with open(path, 'rt') as ark_t_file:
        for line in ark_t_file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            assert len(parts) == 2 \
                and parts[1] == '[', \
                """ wrong Kaldi ARK feature format:
                    expected: <key-of-ftr-file> [
                       found: """ + line
            key = parts[0]
            yield key, MatrixReader(ark_t_file)


class MatrixReader:
    def __init__(self, ark_t_file):
        lines = []
        for line in ark_t_file:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if line[-1] == ']':
                break
        assert lines[-1][-1] == ']', \
            """ wrong Kaldi ARK feature format:
                expected: ]
                   found: """ + line
        lines[-1] = lines[-1][:-1].strip()
        if len(lines[-1]) == 0:
            lines = lines[:-1]
        self.ftrs = np.atleast_2d(np.loadtxt(lines, dtype=np.float32))
        expectNSamples = len(lines)
        self.nSamples, self.nDim = self.ftrs.shape
        if expectNSamples != self.nSamples:
            assert expectNSamples == self.nDim, \
                """ something wrong with features dimension """
            self.ftrs = self.ftrs.transpose()
            self.nSamples, self.nDim = self.ftrs.shape
        self.__returnedVector__ = -1

    def readvec(self):
        self.__returnedVector__ += 1
        return self.ftrs[self.__returnedVector__]

    def getall(self):
        return self.ftrs
