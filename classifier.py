from math import sqrt
from numpy import concatenate

import numpy as np
from pandas import read_csv
from keras.layers import Dropout
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import RepeatVector
from keras.models import model_from_yaml
import os
from keras.utils import plot_model

from keras.utils.vis_utils import plot_model

from keras.utils.vis_utils import model_to_dot
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# sample class bary

data_for_model_training = "onehotforclassifier.csv"

saved_model = "model_single_autoencoder.h5"
malware_class_dir = r"C:\Users\pups1\PycharmProjects\project\Lstmautoencoder_for_malwareapi\finalcsv\Browsefox"
apa = 1
batch = 10




def classifier(samples):
	# create model
	model = Sequential()
	model.add(Dense(150, input_dim=9254, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(150, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(15, activation='softmax'))

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model




dataframe = read_csv(data_for_model_training)

dataset = dataframe.values

print(dataset.shape)



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# decide which features are input which column are output
# in our case the last col indicate which class this sample belongs to
# in our case malware has 15 classes
X = dataset[:,:-15]
Y = dataset[:,-15:]
#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#estimator = KerasClassifier(build_fn=classifier(dataset), epochs=200, batch_size=5, verbose=0)

model = classifier(dataset)
model.fit(X,Y,validation_split=0.1, epochs=10, batch_size=10)

#results = cross_val_score(estimator, X, Y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

'''
for train, test in kfold.split(X, Y):
  # create model
	model = Sequential()
	model = classifier(dataset)
	#model.fit(X, Y, epochs=200, batch_size=10)
	model.fit(X[train], Y[train], validation_split=0.33, epochs=200, batch_size=10, verbose=0)
	# evaluate the model
	scores = model.evaluate(X[test], Y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

'''


