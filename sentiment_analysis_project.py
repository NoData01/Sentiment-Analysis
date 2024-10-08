# -*- coding: utf-8 -*-
"""Sentiment_Analysis_Project.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oAetsn5_r4p633Qz-zm8F7YcS3lEz-Ll
"""

#Edited on 15 June 2022
#Note NLP will be coming out for assessment


import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D

#%% Statics
CSV_URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

#EDA
#%% Step1 - Load Data
df = pd.read_csv(CSV_URL)
df_copy = df.copy() #a copy to prevent wasting time loading the data later

#%% Step2 - Data Inspection/Visualization
df.head(10)
df.tail(10)
df.info() #cuz we deal with string not number
df.describe()

df['sentiment'].unique() #to get the unique target
df['review'][0] #a positive review for index 0
df['sentiment'][0] 

df.duplicated().sum() #There is 418 duplicates
df[df.duplicated()]
#Note that there is a '<br /><br />' in df['review'][0] same goes with [1] and so on,
# <br /> HTML tags have to be removed
# Numbers can be filtered
# Need to remove duplicated data

#%% Step3 = Data Cleaning
# to remove duplicated data
df = df.drop_duplicates()

# to remove html tags

#df['review'][0].replace('<br /> ',' ') # need to do one by one but careful, 
#what if html tags is differ like <h1 / >, then its not working

review = df['review'].values # features of X
sentiment = df['sentiment'].values # target, y

for index,rev in enumerate(review):
  print(index)
  print(rev)
  #remove html tags
  # ? dont be greedy
  # * zero or more occurances
  # Any characterexcept new line (/n)

  review[index] = re.sub('<.*?>',' ',rev)

  # convert into lower case, can do it in data cleaning instead of preprocessing, 
  # remove numbers
  # ^ means NOT
  review[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split() #anything that not a-z and A-Z will be replace by empty set

#%% Step4 - Features Selection
# Nothing to select

#%% Step5 - Data Preprocessing
#           1)Convert into lower case (DONE)
#           2)Tokenization
vocab_size = 10000 # mean 10000 words will get characterized,other than that is not
oov_token = 'OOV'

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review) # to learn all the words
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(review) # to convert into numbers

#           3)Padding & Trunctation
length_of_review = [len(i) for i in train_sequences] #list comprehension
print(np.median(length_of_review)) # to get the number of max length for padding

max_len = 180

padded_review = pad_sequences(train_sequences,maxlen = max_len,
                             padding='post',
                             truncating='post')

#           4)OneHotEncoding for the Target
ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment,axis=-1))

#           5)Train test split
X_train,X_test,y_train,y_test = train_test_split(padded_review,
                                                 sentiment,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1) # since it is in 2 dims, change into 3 dims
X_test = np.expand_dims(X_test,axis=-1)

#%% Model development
#Use LSTM layers,dropout,dense,input
#achieve >90% f1 score
from tensorflow.keras.layers import Bidirectional,Embedding

embedding_dim = 64
model=Sequential()
model.add(Input(shape=(180))) #np.shape(X_train)[1:]  #input_length #features  (it have to be in 3 dims)
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=True)))
# model.add(LSTM(128,return_sequences=True)) #only once (return_sequences=true) when LSTM meet LSTM after
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))#output layer, (Positive,Negative)
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics='acc')

plot_model(model, to_file='model_plot.png', show_shapes=True, 
           show_layer_names=True)

hist = model.fit(X_train,y_train,
                 validation_data=(X_test,y_test),
                 epochs=10,
                 batch_size=128)

#%%
import matplotlib.pyplot as plt
hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--',label='Training acc')
plt.plot(hist.history['val_acc'],label='Validation acc')
plt.legend()
plt.show()


#%% Model evaluation
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

results = model.evaluate(X_test,y_test)
print(results)
y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(model.predict(X_test),axis=1)

cm = confusion_matrix(y_true,y_pred)
cr = classification_report(y_true,y_pred)
print(cm)
print(cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% Model saving
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)

token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
with open(TOKENIZER_PATH,'w') as file:
  json.dump(token_json,file)

OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)

#%% Discussion (what if we cant achieve 95%..then we can describe the data in discussion)

# Discuss your result
# Model achieve around 84% accuracy during training
# Recall and fi score reports 87% and 84% respectively
# However, the model starts to overfit after 2nd epoch
# Early stopping can be introduced in future to prevent overfitting
# Increase dropout rate to control overfitting
# Trying with different DL architecture for example BERT model,transformer
# model,GPT3 model may help to improve the model

# 1) result --> discussion on the results
# 2) give suggestion -->how to improve your model
# 3) gather evidences showing what went wrong during training/model development


#%% Deployment unusually done on PC/mobile phone

# to load trained data
loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))
loaded_model.summary()

# to load tokenizer
with open(TOKENIZER_PATH,'r') as json_file:
  loaded_tokenizer = json.load(json_file)

#%%
# testing the review
input_review = 'This movie so good, the trailer intrigues me to watch.\
                    The movie is funny. I love it so much'
# input_review = input('type your review here')

# preprocessing
input_review = re.sub('<.*?>',' ',input_review)
input_review = re.sub('[^a-zA-Z]',' ',input_review).lower().split()

tokenizer = tokenizer_from_json(loaded_tokenizer)
input_review_encoded = tokenizer.texts_to_sequences(input_review)

input_review_encoded = pad_sequences(np.array(input_review_encoded).T,
                                     maxlen=max_len,
                                     padding='post',
                                     truncating='post')


outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)

print(ohe.inverse_transform(outcome))