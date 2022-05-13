# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:45:32 2022

@author: Fatin
"""
import pandas as pd
from sentiment_analysis_modules import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import datetime


URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
TOKEN_SAVE_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
PATH_LOGS = os.path.join(os.getcwd(), 'logs')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')

#%%
# EDA
# 1: Data Import
df = pd.read_csv(URL)
review = df['review']
sentiment = df['sentiment']

# 2: Data Cleaning
# Remove tags

eda = ExploratoryDataAnalysis()
review = eda.remove_tags(review)
review = eda.lower_split(review)

# 3: Features Selection
# 4: Data Vectorization
review = eda.sentiment_tokenizer(review, TOKEN_SAVE_PATH)
review = eda.sentiment_pad_sequence(review)

# 5: Preprocessing
# One hot encoder
one_hot_encoder = OneHotEncoder(sparse=False)
sentiment = one_hot_encoder.fit_transform(np.expand_dims(sentiment, axis=-1))

# Calculate number of categories
nb_categories = len(np.unique(sentiment))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(review, sentiment, 
                                                    test_size=0.3, random_state=123)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# From here you will know that [0,1] is positive, [1,0] is negative
print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0)))

#%% Model Creation

mc = ModelCreation()

num_words = 10000

model = mc.lstm_layer(num_words, nb_categories)
log_dir = os.path.join(PATH_LOGS,
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%% Compile & Model Fitting

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.fit(X_train, y_train, epochs=10,
          validation_data=(X_test, y_test),
          callbacks=tensorboard_callback)

#%% Model Evaluation
# Append approach

# Preallocation of memory approach

predicted_advanced = np.empty([len(X_test), 2])

for index, test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test, axis=0))

#%% Model Analysis

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()
me.report_metrics(y_true, y_pred)

#%% Model Deployment
model.save(MODEL_SAVE_PATH)