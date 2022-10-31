from unittest import result
import pandas as pd
import numpy as np
import transformers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Softmax,Dense,Activation,Dropout,Conv1D,Flatten,GRU
import tensorflow as tf
from transformers import TFBertForQuestionAnswering
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('tokenizer')
roberta_model=TFBertForQuestionAnswering.from_pretrained('roberta_model')

from transformers import AutoTokenizer,TFAutoModel
max_length_roberta=55
tf.keras.backend.clear_session()
input1 = Input(shape=(max_length_roberta,),name='input_id',dtype=tf.int32)
input2 = Input(shape=(max_length_roberta,),name='attention_mask',dtype=tf.int32)
scores = roberta_model(input1,attention_mask=input2)
dropout1=Dropout(0.2)(scores.hidden_states[-1])
dropout2=Dropout(0.2)(scores.hidden_states[-1])
conv1=Conv1D(8,1)(dropout1)
conv2=Conv1D(8,1)(dropout2)
dropout3=Dropout(0.2)(conv1)
dropout4=Dropout(0.2)(conv2)
conv3=Conv1D(1,1)(dropout3)
conv4=Conv1D(1,1)(dropout4)

flatten1=Flatten()(conv3)
flatten2=Flatten()(conv4)
softmax1 = Activation('softmax')(flatten1)
softmax2 = Activation('softmax')(flatten2)
model = Model(inputs=[input1,input2],outputs=[softmax1,softmax2])
model.load_weights('BERTmodel.hdf5')

def jaccard(x,y):
    str1=x
    str2=y 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0):
        return 0.5
    c = a.intersection(b)
    
    return float(len(c)) / (len(a) + len(b) - len(c))


def function_1(x):
  max_length_roberta=55
  input_id=np.zeros((1,max_length_roberta))
  mask=np.zeros((1,max_length_roberta))
  text=x[0]
  sentiment=x[1]
  encoded=tokenizer.encode_plus(sentiment,text,max_length=max_length_roberta,return_attention_mask=True,padding='max_length',add_special_tokens=True,return_tensors='tf')
  input_id[0]=encoded['input_ids'][0][:max_length_roberta]
  mask[0]=encoded['attention_mask'][0][:max_length_roberta]
  test_input=(input_id,mask)
  start_test_pred,end_test_pred=model.predict(test_input)
  start_pred=np.argmax(start_test_pred,axis=-1)
  end_pred=np.argmax(end_test_pred,axis=-1)
  encoded=tokenizer.encode_plus(sentiment,text,max_length=max_length_roberta,return_attention_mask=True,padding='max_length',add_special_tokens=True,return_tensors='tf')
  pred_text=tokenizer.decode(encoded['input_ids'][0][start_pred[0]:end_pred[0]+1])
  return pred_text

def function_2(x):
  max_length_roberta=55
  input_id=np.zeros((1,max_length_roberta))
  mask=np.zeros((1,max_length_roberta))
  text=x[0]
  sentiment=x[1]
  actual_pred=x[2]
  encoded=tokenizer.encode_plus(sentiment,text,max_length=max_length_roberta,return_attention_mask=True,padding='max_length',add_special_tokens=True,return_tensors='tf')
  input_id[0]=encoded['input_ids'][0][:max_length_roberta]
  mask[0]=encoded['attention_mask'][0][:max_length_roberta]
  test_input=(input_id,mask)
  start_test_pred,end_test_pred=model.predict(test_input)
  start_pred=np.argmax(start_test_pred,axis=-1)
  end_pred=np.argmax(end_test_pred,axis=-1)
  encoded=tokenizer.encode_plus(sentiment,text,max_length=max_length_roberta,return_attention_mask=True,padding='max_length',add_special_tokens=True,return_tensors='tf')
  pred_text=tokenizer.decode(encoded['input_ids'][0][start_pred[0]:end_pred[0]+1])
  jcd_score=jaccard(actual_pred,pred_text)
  return jcd_score


from flask import Flask,redirect,url_for,render_template,request
app=Flask(__name__,'',template_folder='templates',static_folder='static')

@app.route('/')
def home():
    return render_template('home_page.html')

@app.route('/prediction',methods=['POST','GET'])
def process():
    if request.method=='POST':
        text=request.form['text_sent']
        sentiment=request.form['sentiment']
        output=function_1((text,sentiment.lower()))
    return render_template('prediction_page.html',result=output)

@app.route('/score',methods=['POST','GET'])
def score():
    if request.method=='POST':
        text=request.form['text_sent']
        sentiment=request.form['sentiment']
        act_sent_text=request.form['act_sent']
        output=function_2((text,sentiment.lower(),act_sent_text))
    return render_template('score_page.html',result=output)
if __name__=='__main__':
    app.run(host='0.0.0.0',port='0')