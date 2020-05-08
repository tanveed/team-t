# General Libraries
import json
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# NLP
import nltk
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer


# ML/DL
import tensorflow as tf
import pickle


from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, MaxPooling1D, LSTM, BatchNormalization, SpatialDropout1D, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text as txt
from keras.preprocessing import sequence
from keras import utils
from keras import regularizers
from keras.models import load_model

# Define some regex patterns for cleaning
nltk.download('stopwords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def eval(df, models):
	# This is where you call your model to get the number of stars output
	# Pre-process are text
	X = df['text'].fillna('').values
	
	# Load tokenizer
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	# Load models
	baseline, lstm, lstm_1, lstm_2, lstm_3, lstm_4, lstm_5 = models

	# Vectorize/Tokenizer text
	X_baseline = tokenizer.texts_to_matrix(X)
	X_lstm = tokenizer.texts_to_sequences(X)
	X_lstm = pad_sequences(X_lstm, maxlen=400)
	
	# EVALUATE
	cols = [1, 2, 3, 4, 5]

	# Baseline
	baseline_preds = pd.DataFrame(baseline.predict(X_baseline), columns=cols)
	baseline_preds['baseline_pred'] = baseline_preds.idxmax(axis=1)

	# LSTM
	lstm_preds = pd.DataFrame(lstm.predict(X_lstm), columns=cols)
	lstm_preds['lstm_pred'] = lstm_preds.idxmax(axis=1)

	# One vs. All
	one_star_ps = lstm_1.predict(X_lstm)
	two_star_ps = lstm_2.predict(X_lstm)
	three_star_ps = lstm_3.predict(X_lstm)
	four_star_ps = lstm_4.predict(X_lstm)
	five_star_ps = lstm_5.predict(X_lstm)

	data = [one_star_ps.flatten(), two_star_ps.flatten(), three_star_ps.flatten(), four_star_ps.flatten(), five_star_ps.flatten()]
	cols = [1, 2, 3, 4, 5]
	ova_preds = pd.DataFrame(data=data, index=cols).T

	ova_preds["ova_pred"] = ova_preds.idxmax(axis=1)

	all_preds = pd.DataFrame([baseline_preds['baseline_pred'], lstm_preds['lstm_pred'], ova_preds['ova_pred']]).T
	all_preds["final_pred"] = all_preds.mode(axis=1)[0]
	return all_preds["final_pred"]

def adjust_stopwords(stopwords):
    words_to_keep = set('nor', 'not', 'very', 'no')

def clean_text(text):
    new_text = BeautifulSoup(text, "lxml").text # HTML decoding
    new_text = new_text.lower() # lowercase text
    new_text = REPLACE_BY_SPACE_RE.sub(' ', new_text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    new_text = BAD_SYMBOLS_RE.sub(' ', new_text) # delete symbols which are in BAD_SYMBOLS_RE from text
   
    ps = PorterStemmer()
    
    new_text = ' '.join(ps.stem(word) for word in new_text.split()) # keeping all words, no stop word removal
#     new_text = ' '.join(ps.stem(word) for word in new_text.split() if word not in STOPWORDS) # delete stopwords from text and stem
    return new_text

def load_models():
	# Optimizer
	lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=.001, decay_steps=10000, decay_rate=0.9)
	optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.99, amsgrad=False, clipvalue=.3)

	# Baseline
	baseline = load_model('./models/baseline.h5')

	baseline.compile(loss='categorical_crossentropy',
				optimizer=optimizer,
				metrics=['accuracy'])

	# LSTM
	lstm = load_model('./models/lstm.h5')

	lstm.compile(loss='categorical_crossentropy',
				optimizer=optimizer,
				metrics=['accuracy'])


	# One vs. all
	lstm_1 = load_model('./models/one_star.h5')

	lstm_1.compile(loss='binary_crossentropy',
					optimizer=optimizer,
					metrics=['accuracy'])

	lstm_2 = load_model('./models/two_star.h5')

	lstm_2.compile(loss='binary_crossentropy',
					optimizer=optimizer,
					metrics=['accuracy'])

	lstm_3 = load_model('./models/three_star.h5')

	lstm_3.compile(loss='binary_crossentropy',
					optimizer=optimizer,
					metrics=['accuracy'])

	lstm_4 = load_model('./models/four_star.h5')

	lstm_4.compile(loss='binary_crossentropy',
					optimizer=optimizer,
					metrics=['accuracy'])

	lstm_5 = load_model('./models/five_star.h5')

	lstm_5.compile(loss='binary_crossentropy',
					optimizer=optimizer,
					metrics=['accuracy'])

	return (baseline, lstm, lstm_1, lstm_2, lstm_3, lstm_4, lstm_5)

if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	val_df = pd.read_json(validation_file, lines = True)
	
	# Clean text
	val_df['text'] = val_df['text'].apply(clean_text)

	# Evaluate
	models = load_models()
	preds = eval(val_df, models)

	# Add final predictions
	val_df['predicted_stars'] = preds

	# Write predictions
	val_df[['review_id', 'predicted_stars']].to_json('output.jsonl')
	print("Output prediction file written")

	# with open("output.jsonl", "w") as fw:
	# 	with open(validation_file, "r") as fr:
	# 		for line in fr:
	# 			review = json.loads(line)
	# 			fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'], models)})+"\n")
	# print("Output prediction file written")
else:
	print("No validation file given")