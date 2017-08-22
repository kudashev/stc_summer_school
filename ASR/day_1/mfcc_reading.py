#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io.wavfile
from python_speech_features import mfcc


"""
1. Извлечение признаков из звукового файла и их запись в бинарник:
(можно вставить дефолтные значения параметров для этого ридера mfcc)

"""

wav_name = 'example.wav'

# чтение отсчетов wav файла:
s_rate, samples = scipy.io.wavfile.read(wav_name)

# чтение MFCC признаков по отсчетам файла:
features = mfcc(samples, s_rate)

# запись признаков в бинарник:
features_file = wav_name.split('.')[0]
np.save(features_file, features)

# запись признаков в формат ark,t:
file_name = features_file + '.txtftr'
with open(file_name, 'w') as fn:
    fn.write('{} [\n'.format(file_name))
    for ftr in features:
        fn.write(str(ftr).replace('\n', '')[1:-1] + '\n')
    fn.write(']')




















