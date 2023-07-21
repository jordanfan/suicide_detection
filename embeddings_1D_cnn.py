import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns  # for nicer plots
sns.set(style="darkgrid")  # default style
import plotly.graph_objs as plotly  # for interactive plots
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA

import nltk
import gensim
import pickle

from collections import Counter
import warnings 
warnings.filterwarnings("ignore")



def train_embedding_1Dcnn(vocab_size, embedding_output_dim, X_train_reduced, y_train, cnn_filters_kernel = [], 
	epochs = 10, random_state = 1234, model_fn = "", history_fn = ""):
	'''
	Function to generalize embedding training, allowing different parameters to be passed in 
	to try training embeddings model

	Inputs:
		vocab_size (int): size of total vocabulary
		embedding_output_dim (int): size of the embedding vector to output from Embedding layer 
		X_train_reduced (array): training data to train model, should be data that is 
								padded, encoded, and reduced down to vocab size 
		y_train (array): test data to train model
		cnn_filters_kernel(list): list of tuples for each new layer in CNN neural network 
								first argument of tuple is the number of filters in 1D CNN
								second arguemnt of tuple is the kernel size in 1D CNN 
		epochs (int): number of training iterations 
		random_state (int): set random state for reproducibility
		model_fn (str): filename to save the resulting model, model will be saved as keras model 
		history_fn (str): filename to save the resulting history, history will be saved as pkl file 
	'''
	tf.keras.backend.clear_session()
	tf.random.set_seed(random_state)
	#Add early stopping to prevent extreme overfitting 
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights = True) 

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Embedding(input_dim = vocab_size + 2, #Add 2 for padding and unknown tokens
										output_dim = embedding_output_dim))

	for i in range(len(cnn_filters_kernel)):
		num_filters = cnn_filters_kernel[i][0]
		kernel_size = cnn_filters_kernel[i][1]
		model.add(tf.keras.layers.Conv1D(filters = num_filters,
										kernel_size = kernel_size,
										padding = "same",
										activation = "relu"))
		model.add(tf.keras.layers.Dropout(0.5))
		model.add(tf.keras.layers.MaxPooling1D(pool_size = 2))

	model.add(tf.keras.layers.GlobalAveragePooling1D())

	model.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))

	model.compile(loss = "binary_crossentropy",
					optimizer = "adam", 
					metrics = ["accuracy"])
	tf.random.set_seed(random_state)
	np.random.seed(random_state)
	history = model.fit(
		x = X_train_reduced,
		y = y_train,
		validation_split=0.2, 
		epochs=epochs, 
		callbacks=[callback])

	model.save(model_fn)
	with open(history_fn, "wb") as handle: 
		pickle.dump(history, handle)

if __name__ == "__main__":

	with open('X_train_reduced.pkl', 'rb') as handle:
		X_train_reduced = pickle.load(handle)
	with open('X_val_reduced.pkl', 'rb') as handle:
		X_val_reduced = pickle.load(handle)
	with open('X_test_reduced.pkl', 'rb') as handle:
		X_test_reduced = pickle.load(handle)
	    
	with open("y_train.pkl", 'rb') as handle:
		y_train = pickle.load(handle)
	with open("y_val.pkl", 'rb') as handle:
		y_val = pickle.load(handle)
	with open("y_test.pkl", 'rb') as handle:
		y_test = pickle.load(handle)


	train_embedding_1Dcnn(4843, 100, X_train_reduced, y_train, 
						cnn_filters_kernel = [(128, 4), (64, 4), (32, 5)], 
						epochs = 10, random_state = 5234, 
						model_fn = "model_cnn_1.keras", history_fn = "history_cnn_1.pkl") 

	train_embedding_1Dcnn(4843, 100, X_train_reduced, y_train, 
						cnn_filters_kernel = [(64, 4), (32, 4)], 
						epochs = 10, random_state = 5234, 
						model_fn = "model_cnn_2.keras", history_fn = "history_cnn_2.pkl")

	train_embedding_1Dcnn(4843, 100, X_train_reduced, y_train, 
						cnn_filters_kernel = [(256, 4), (128, 4), (64, 5)], 
						epochs = 10, random_state = 5234, 
						model_fn = "model_cnn_3.keras", history_fn = "history_cnn_3.pkl")





