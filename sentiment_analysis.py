# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:45:39 2022

@author: Fatin
"""
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer 
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import datetime



class ExploratoryDataAnalysis():
    
    def __init__(self):
        '''
        

        Returns
        -------
        None.

        '''
        pass
    
    def remove_tags(self, data):
        '''
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        '''
        for index, text in enumerate(data):
            data[index] = re.sub('<.*?>', '', text)
        return data
    
    def lower_split(self, data):
        '''
        This function converts all letters into lowercase
        Also filter

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        data: List
            Cleaned data with all letters converter into lowercase and split
        '''
        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z', ' ', text).lower().split()
        
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

        return pad_sequences(review_dummy, maxlen=200, padding='post,
                      truncating='post')
    
    
class ModelCreation():
    
    def __init__(self):
        pass
    
    def lstm_layer(self, num_words, embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model

    def simple_lstm_layer(self)

        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model
            
    def rnn_layer(self, num_words, nb_categories):
        
        
        
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
    test = eda.sentiment_tokenizer(test, token_save_path=TOKENIZER_JSON_PATH) # TOkenize
    test = eda.sentiment_pad_sequence(test)
    
    #%%
    nb_categories = len(sentiment.unique())
    mc = ModelCreation()
    model = mc.simple_lstm_layer(10000, nb_categories)
    
    # tab --> make indentation
    # shift+tab --> reverse the indentation
    #%%
    
    # 3: Data Cleaning
    for index, text in enumerate(review_dummy):
        review_dummy[index] = re.sub('<.*?>', '', text)
    
    # to convert to lowercase and split 
    for index, text in enumerate(review_dummy):
        review_dummy[index] = re.sub('[^a-zA-Z', ' ', text).lower().split()
        
    
    # 5: Data Preprocessing
    # tokenize for review
    
    num_words = 10000
    oov_token = '<OOV>'
    
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(review_dummy)
    
    # To observe the number of words
    word_index = tokenizer.word_index
    print(word_index)
    print(dict(list(word_index.items())[0:10]))
    
    # To vectorize the sequences of text
    review_dummy = tokenizer.texts_to_sequences(review_dummy)
    
    temp = [np.shape(i) for i in review_dummy]
    
    np.mean(temp)
    
    review_dummy = pad_sequences(review_dummy, maxlen=200,
                                 padding='post', truncating='post')
    
    # One hot encoding for label
    one_hot_encoder = OneHotEncoder(sparse=False)
    sentiment_encoded = one_hot_encoder.fit_transform(np.expand_dims(sentiment_dummy, axis=-1))
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(review_dummy,
                                                        sentiment_encoded,
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)No docume                                                    test_size=0.3,
                                                        random_state=123)
    #%% Model Creation
    
    model = Sequential()
    model.add(LSTM(128, input_shape(np.shape(X_train)[1:])), return_sequences=True)
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    
    #%% Callbacks
    
    log_dir = os.path.join(PATH_LOGS, datetime.datetime.now(),
                           strftime('%Y%m%d-%H%M%S'))
    
    tensorback_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    #%% Compile & model fitting
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics='acc')
    
    model.fit(X_train, y_train, epochs=5,
              validation_data=(X_test, y_test), callbacks=tensorback_callback)
    
    #%% Model Evaluation
    # Append approach
    
    predicted = []
    
    for test in X_test:
        predicted.append(model.predict(np.expand_dims(test, axis-0)))
        
    # Preallocation of memory approach
    
    predicted_advanced = np.empty([len(X_test), 2])
    
    for index, test in enumerate(X_test):
        predicted_advanced[index,:] = model.predict(np.expand_dims(test, axis=0))
    
    
    #%% Model Analysis
    
    y_pred = np.argmax(predicted_advanced, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    
    #%% Model Deployment
    model.save(MODEL_SAVE_PATH)