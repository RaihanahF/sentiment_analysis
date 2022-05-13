# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:18:36 2022

@author: Fatin
"""

import re
from tensorflow.keras.preprocessing.text import Tokenizer
import json
#from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional
#from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#import datetime
#import numpy as np

class ExploratoryDataAnalysis():
    
    def __init__(self):
        
        pass
    
    def remove_tags(self, data):
        
        for index, text in enumerate(data):
            data[index] = re.sub('<.*?>', '', text)
        return data
    
    def lower_split(self, data):
        
        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()
        
        return data

    def sentiment_tokenizer(self, data, token_save_path, 
                            num_words=10000, oov_token='<OOV>', prt=False):
        # tokenizer to vectorize the words
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        
        # to save the tokenizer for deployment purpose
        token_json = tokenizer.to_json()
        
        with open(token_save_path, 'w') as json_file:
            json.dump(token_json, json_file)
            
        # to observe the number of words
        word_index = tokenizer.word_index
        
        if prt == True:
            # to view the tokenized words
            # print(word_index)
            print(dict(list(word_index())[0:10]))
        
        # to vectorize the sequence of text
        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def sentiment_pad_sequence(self, data):

        return pad_sequences(data, maxlen=200, padding='post', truncating='post')
    
    
class ModelCreation():
    
    def __init__(self):
        pass
    
    def lstm_layer(self, num_words, nb_categories, 
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model

    def simple_lstm_layer(self, num_words, nb_categories,
                          embedding_output=64, nodes=32, dropout=0.2):

        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model
            
class ModelEvaluation():
    def report_metrics(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))
        
#%%
if __name__ == '__main__':
    
    import os
    import pandas as pd
    
    PATH_LOGS = os.path.join(os.getcwd(), 'logs')
    MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')
    TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
    
    URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
    
    df = pd.read_csv(URL)
    review = df['review']
    sentiment = df['sentiment']
    
    #%%
    eda = ExploratoryDataAnalysis()
    test = eda.remove_tags(review) # remove tags
    test = eda.lower_split(test) # convert to lowercase and split
    test = eda.sentiment_tokenizer(test, token_save_path=TOKENIZER_JSON_PATH) # Tokenize to vectorize the words
    test = eda.sentiment_pad_sequence(test)
    
    #%%
    nb_categories = len(sentiment.unique())
    mc = ModelCreation()
    model = mc.simple_lstm_layer(10000, nb_categories)
    
# tab --> make indentation
# shift+tab --> reverse the indentation

