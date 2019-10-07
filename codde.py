'''
The multi-input network is created with two auxiliary inputs and one main input. The main input is the text data(clinical evidence)
to the birectional LSTM. Two auxiliary inputs carry the gene and variants information. 
'''

import numpy as np
import pandas as pd

#Reading the inputs
train_var = pd.read_csv("F:/to do/kaggle/personalized medicine/training_variants/training_variants")
train_var.head(10)
test_var=pd.read_csv("F:/to do/kaggle/personalized medicine/stage2_test_variants.csv/stage2_test_variants.csv")

train_text=pd.read_csv("F:/to do/kaggle/personalized medicine/training_text/training_text",sep="\|\|",engine='python',
                       header=None,skiprows=1,names=["ID","text"])

frames=[train_var,test_var]
tot_var=pd.concat(frames)


print(len(train_text["text"]))
#v_size=0
#for i in range(1,len(train_text["text"])):
 #   a=len(train_text["text"][i])
 #   v_size=v_size+a 

#One hot representation of labels
from keras.utils import to_categorical
class_labels=pd.read_csv("F:/to do/kaggle/personalized medicine/training_variants/training_variants",skiprows=1,
                         usecols=[3],names=["class"])

classes=to_categorical(class_labels,num_classes=10)
classes=classes[:,1:]


from keras.preprocessing.text import Tokenizer,one_hot
from keras.preprocessing.sequence import pad_sequences,skipgrams

from keras.models import Model
from keras.layers import Embedding,LSTM,Dense,Dropout,concatenate,Activation,Input
from keras.layers.wrappers import Bidirectional
from keras import backend as K
#from keras.layers.core import Lambda

#Training texts(Main Input) are converted to sequences, and zero padded to fit to specified length.
tokenizer=Tokenizer(num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ",char_level=False)
tokenizer.fit_on_texts(train_text["text"][0:])
train_seq=tokenizer.texts_to_sequences(train_text["text"][0:])
train=tokenizer.sequences_to_matrix(train_seq,mode='tfidf')
word_index=tokenizer.word_index
pad=pad_sequences(train, maxlen=500, dtype='int32',padding='post', truncating='post', value=0)

#Auxiliary inputs are converted to matrices and normalized to unit Norm.
token_gene=Tokenizer(num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ",char_level=False)
token_gene.fit_on_texts(tot_var["Gene"][0:])
gene_seq=token_gene.texts_to_matrix(tot_var["Gene"][0:],mode='count')
#gene_pad=pad_sequences(gene_seq, maxlen=50, dtype='int32',padding='pre', truncating='pre', value=0)
token_var=Tokenizer(num_words=None,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ",char_level=False)
token_var.fit_on_texts(train_var["Variation"][0:])
var_seq=token_var.texts_to_matrix(tot_var["Variation"][0:],mode='count')
#var_pad=pad_sequences(var_seq, maxlen=50, dtype='int32',padding='pre', truncating='pre', value=0)

#aux1=np.column_stack((gene_seq,var_seq))
#aux=np.reshape(aux,(26568,1))
aux_sh1=np.shape(var_seq)[1]
aux_sh2=np.shape(gene_seq)[1]
from sklearn.preprocessing import Normalizer 
normalizer1=Normalizer().fit(pad)
pad=normalizer1.transform(pad)
normalizer2=Normalizer().fit(var_seq)
aux_1=normalizer2.transform(var_seq)
normalizer3=Normalizer().fit(gene_seq)
aux_2=normalizer2.transform(gene_seq)

#embeddings_index = {}
#f = open(os.path.join('F:/to do/kaggle/personalized medicine/glove.6B', 'glove.6B.50d.txt'),encoding="utf8")
#for line in f:
 #   values = line.split()
  #  word = values[0]
   # coefs = np.asarray(values[1:], dtype='float32')
    #embeddings_index[word] = coefs
#f.close()

#print('Found %s word vectors.' % len(embeddings_index))

#embedding_matrix = np.zeros((len(word_index) + 1, 50))
#for word, i in word_index.items():
 #   embedding_vector = embeddings_index.get(word)
  #  if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
   #     embedding_matrix[i] = embedding_vector

#from keras.models import load_model

#emb_model=load_model('F:/to do/kaggle/personalized medicine/emb.h5')
#wt=emb_model.layers[0].get_weights()
#wt=np.reshape(wt,(197072,50))
#wt=wt[0:181264,:]
#wt=np.reshape(wt,(1,181264,50))

from gensim.models import Word2Vec

#Loading the trained Word2Vec model(Refer emb.py file for creating the embedding model)
word2vec = Word2Vec.load("F:/to do/kaggle/personalized medicine/embed/emb_nw.txt")

#Embedding matrix is created for the embedding layer using the the above model.
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    if word in word2vec.wv.vocab:
        embedding_matrix[i] = word2vec.wv.word_vec(word)
#print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#Embedding layer to be used in the network
embedding_layer = Embedding(len(word_index) + 1,100,weights=[embedding_matrix],input_length=500,trainable=False)        
#embedding_layer = Embedding(len(word_index) + 1,50,input_length=400)

#pad=np.reshape(pad,(1328400,1))
#skip=skipgrams(train_seq,v_size,window_size=7, negative_samples=1., shuffle=True,categorical=False, sampling_table=None)
print(np.shape(pad))
#print(np.shape(skip)) 
#inp=K.constant(pad)
#aux1=K.constant(aux)
#print(K.shape(pad))
#print(K.shape(aux))
#def bottom(x):
#     x=Dense(16,activation='relu')(x)
#     out=Dense(9,activation='softmax')(x)
#     return out    

#def shape(x):
#    return np.shape(x)

#cus_layer=Lambda(bottom,output_shape=shape)

# Creation of the network and training
inp=Input(shape=(500,),name='main')
a=embedding_layer(inp)
b=Bidirectional(LSTM(25,activation='relu'),merge_mode='concat')(a)
#b=Bidirectional(LSTM(8,activation='relu'),merge_mode='concat')(b)
aux_in1=Input(shape=(aux_sh1,),name='aux1')
#x=K.concatenate([b,aux1],axis=1)
x=concatenate([b,aux_in1])
#x=K.dropout(x,0.1)
#out=cus_layer(x)
x=Dense(250,activation='relu')(x)
#x=Dense(50,activation='relu')(x)
#x=K.relu(x)
aux_in2=Input(shape=(aux_sh2,),name='aux2') 
y=concatenate([x,aux_in2])
y=Dense(50,activation='relu')(y)
y=Dense(9,activation='relu')(y)
out=Dense(9,activation='softmax')(y)
#out=K.softmax(x)
model=Model(inputs=[inp,aux_in1,aux_in2],outputs=out)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit([pad,aux_1,aux_2],classes,epochs=6,batch_size=32,shuffle=True,validation_split=0.1)
