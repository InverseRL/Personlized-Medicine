# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:31:22 2017

@author: Jeshurun
"""

import numpy as np
import pandas as pd

train_text=pd.read_csv("F:/to do/kaggle/personalized medicine/training_text/training_text",sep="\|\|",engine='python',
                       header=None,skiprows=1,names=["ID","text"])

#test_text=pd.read_csv("F:/to do/kaggle/personalized medicine/test_text/test_text",sep="\|\|",engine='python' header=None,skiprows=1,names=["ID","text"])
s2_test_text=pd.read_csv("F:/to do/kaggle/personalized medicine/stage2_test_text.csv/stage2_test_text.csv",sep="\|\|",engine='python',
                       header=None,skiprows=1,names=["ID","text"])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences,skipgrams
from keras.layers import Embedding
from keras.models import Sequential

frames=[train_text,s2_test_text]
tot_text=pd.concat(frames)


 
from gensim.models import Word2Vec
 
model=Word2Vec(tot_text["text"][0:],min_count=2,size=50,sg=1,window=3)
model.save("F:/to do/kaggle/personalized medicine/embed/emb_50.txt")


#model=Sequential()
#model.add(Embedding(len(word_index)+1,50,input_length=400))
#model.compile(loss='mse',optimizer='adam')
#model.predict(pad)

#wt=model.layers[0].get_weights()