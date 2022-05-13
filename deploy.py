# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:20:24 2022

@author: Fatin
"""

from tensorflow.keras.models import load_model
import os
import json
from sentiment_analysis import ExploratoryDataAnalysis
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
#import warnings
#warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')
JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')

#%% Model Loading
sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

#%% Tokenizer Loading
with open(JSON_PATH, 'r') as json_file:
    token = json.load(json_file)
    
#%% EDA

# 1: Load Data
new_review = ['<br \> I decided to watch the movie during my off day from work\
              but I was bored halfway trough the movie so it feels like\
              it has wasted my precious holiday.<br \>']
              
#new_review = [input('Review about the movie\n')]
              
# 2: Clean Data
eda = ExploratoryDataAnalysis()
removed_tags = eda.remove_tags(new_review)
cleaned_input = eda.lower_split(removed_tags)

# 3: Features Selection
# 4: Data Preprocessing

# Vectorize the new review
# Feed the token into keras
loaded_tokenizer = tokenizer_from_json(token)

# Vectorize the review into integers
new_review = loaded_tokenizer.texts_to_sequences(cleaned_input)
new_review = eda.sentiment_pad_sequence(new_review)

# Model Prediction
outcome = sentiment_classifier.predict(np.expand_dims(new_review, axis=-1))

# positive = [0,1]
# negative = [1,0]
sentiment_dict = {1: 'positive', 0:'negative'}
print('this review is:' + sentiment_dict[np.argmax(outcome)])

