# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 18:40:27 2017

@author: Jeshurun
"""

import numpy as np
import pandas as pd

train_var = pd.read_csv("F:/to do/kaggle/personalized medicine/training_variants/training_variants")

train_text=pd.read_csv("F:/to do/kaggle/personalized medicine/training_text/training_text",sep="\|\|",engine='python',
                       header=None,skiprows=1,names=["ID","text"])



print(len(train_text["text"]))
v_size=0
for i in range(1,len(train_text["text"])):
    a=len(train_text["text"][i])
    v_size=v_size+a

from keras.utils import to_categorical
class_labels=pd.read_csv("F:/to do/kaggle/personalized medicine/training_variants/training_variants",skiprows=1,
                         usecols=[3],names=["class"])

classes=to_categorical(class_labels,num_classes=10)
classes=classes[:,1:]
 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences,skipgrams

tokenizer=Tokenizer(num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ",char_level=False)
tokenizer.fit_on_texts(train_text["text"][0:])
train_seq=tokenizer.texts_to_sequences(train_text["text"][0:])
pad=pad_sequences(train_seq, maxlen=256, dtype='int32',padding='pre', truncating='pre', value=0.)

from keras.models import load_model
 
model=load_model('F:/to do/kaggle/personalized medicine/codde_256.h5')
model.fit(pad,classes,epochs=3,batch_size=16,shuffle=False,validation_split=0.1)