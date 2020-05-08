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

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, MaxPooling1D, LSTM, BatchNormalization, SpatialDropout1D, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras import utils
from keras import regularizers
from keras.models import load_model

# Define some regex patterns for cleaning
nltk.download('stopwords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def eval(text):
	# This is where you call your model to get the number of stars output
	text = pre_process(text)
	return 1.0

def pre_process(text):
	cleaned_text = clean_text(text)
	print(cleaned_text)
	return cleaned_text

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
   
    ps = PorterStemmer()
    
    text = ' '.join(ps.stem(word) for word in text.split() if word not in STOPWORDS) # delete stopwords from text and stem
    return text

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

models = load_models()


if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			for line in fr:
				review = json.loads(line)
				fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")