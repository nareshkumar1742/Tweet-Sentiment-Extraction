from unittest import result
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Input,Embedding,GRU,Dense,Flatten,Concatenate,Dropout,LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

tokenizer_text=joblib.load('tokenizer_text.pkl')
tokenizer_sentiment=joblib.load('tokenizer_sentiment.pkl')


def preprocess(sentence):
    sentence=sentence.replace('****',"curse")# changing bad words(marked as **** in text) to 'curse'
    sentence=' '.join(e.lower() for e in sentence.split())
    return sentence.strip()    

def prob_to_binary(x,threshold=0.5):
  lst=[]
  for i in x:
    if i>=threshold:
      lst.append(1)
    else:
      lst.append(0)
  return lst

def pred_text(x):
    pred_array=x[0]
    text=x[1]
    text_list=x[1].split()
    max_len_list=len(text_list)
    indices=np.where(pred_array==1)[0]
    indices=[ind for ind in indices if ind<max_len_list]
    pred_text_list=np.array(text_list)[indices]
    pred_text=' '.join(pred_text_list)
    return pred_text

vocab_size_text=38689
max_length=33
max_length_sentiment=1
vocab_size_sentiment=5

tf.keras.backend.clear_session()
input1=Input(shape=(max_length,),name='input_text')
embed = Embedding(vocab_size_text,200,input_length=max_length,name='embedding',
                      trainable=False)(input1)

input2=Input(shape=(max_length_sentiment,),name='input_sentiment')
embed2=Embedding(vocab_size_sentiment,200,input_length=max_length_sentiment,trainable=False,
                 name='embedding_sentiment')(input2)
concat1=Concatenate(axis=1)([embed,embed2])
gru_1=GRU(100,name='GRU_1',return_sequences=True)(concat1)
gru_2=GRU(32,name='GRU_2',return_sequences=True)(gru_1)
gru_3=GRU(16,name='GRU_3',return_sequences=True)(gru_2)
f1=Flatten()(gru_3)

f1=tf.expand_dims(f1,1)
dense2=Dense(256,activation='relu',kernel_regularizer=l2(0.0001))(f1)
drop1 = Dropout(0.2)(dense2)
ln1= LayerNormalization()(drop1)
dense3=Dense(128,activation='relu',kernel_regularizer=l2(0.0001))(ln1)
drop2 = Dropout(0.2)(dense3)
ln2= LayerNormalization()(drop2)
dense4=Dense(64,activation='relu',kernel_regularizer=l2(0.0001))(ln1)
output=Dense(33,activation='sigmoid',name='output')(dense4)

model=Model(inputs=[input1,input2],outputs=[output])
model.load_weights('dl_base_model.h5')

def jaccard(x,y):
    str1=x
    str2=y 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0):
        return 0.5
    c = a.intersection(b)
    
    return float(len(c)) / (len(a) + len(b) - len(c))


def final_func_1(x):
    text=x[0]
    sentiment=x[1].lower()
    text=preprocess(text)
    text_token=tokenizer_text.texts_to_sequences([text])
    text_padding=pad_sequences(text_token,max_length,padding='post')
    sentiment_token=tokenizer_sentiment.texts_to_sequences([sentiment])
    prediction=model.predict((text_padding,np.array(sentiment_token)))
    prediction=np.squeeze(prediction,1)
    prediction=np.array(prob_to_binary(prediction[0]))
    pred_text_indiv=pred_text((prediction,text))
    return pred_text_indiv

def final_func_2(x):
    text=x[0]
    sentiment=x[1].lower()
    actual_text=x[2]
    text=preprocess(text)
    text_token=tokenizer_text.texts_to_sequences([text])
    text_padding=pad_sequences(text_token,max_length,padding='post')
    sentiment_token=tokenizer_sentiment.texts_to_sequences([sentiment])
    prediction=model.predict((text_padding,np.array(sentiment_token)))
    prediction=np.squeeze(prediction,1)
    prediction=np.array(prob_to_binary(prediction[0]))
    pred_text_indiv=pred_text((prediction,text))
    jcd_score=jaccard(actual_text,pred_text_indiv)
    return jcd_score
