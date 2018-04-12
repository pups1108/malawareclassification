#coding=utf-8

from math import sqrt
import matplotlib.pyplot as plt
from numpy import concatenate
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from keras.models import load_model
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.layers.convolutional import Conv2D
from os import listdir
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import RepeatVector
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import model_from_yaml
import os
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
import random


apa = 200
batch = 10
timestep = 16
numfeature = 25
filenamelist =  r"/Users/user/Desktop/hooklog2"


def gothrougheveryfile(dir):  # go through every csv and concat to one csv this part has move to concatallcsv
    i = 0
    filenamelist = list()
    for filename in os.listdir(dir):  # outside each folder
        wholefilepath = dir + "//" + filename
        filenamelist.append(wholefilepath)
    return filenamelist


def data_to_reconstruction_problem(data, timestep):
    df = DataFrame(data)
    list_concat = list()
    for i in range(timestep - 1, -1, -1):
        tempdf = df.shift(i)
        list_concat.append(tempdf)
    data_for_autoencoder = concat(list_concat, axis=1)
    data_for_autoencoder.dropna(inplace=True)
    return data_for_autoencoder


def out_put_core(writting_list, outaddress, filename):
    thefile = open(outaddress + "\\" + filename + ".txt", 'w')
    for item in writting_list:
        thefile.write("%s\n" % item)


def data_preprocess(file_name, timestep):
    # read
    dataset = read_csv(file_name, header=None, index_col=None)
    if dataset.shape[0]<16:
        return 0,0
    values = dataset.values
    reframed = data_to_reconstruction_problem(values, timestep)
    reframedvalues = reframed
    reframed = reframed.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(reframed)
    dfscaled = DataFrame(scaled)
    valuescaled =dfscaled.values
    return  valuescaled,scaler,reframedvalues


def SingleFileLstmAutoencoder(apa, batch, n_apis, n_features, data_for_model_training):
    #error_list = list()
    W_Hidden1_list = list()
    W_Hidden2_list = list()
    W_Hidden3_list = list()
    W_Hidden4_list = list()
    W_Hidden5_list = list()
    train_X, scaler, y = data_preprocess(data_for_model_training, n_apis)#146,400
    train_X = train_X.reshape((train_X.shape[0], n_apis, n_features))
    sample_number = train_X.shape[0]
    outputlayer2 = n_apis #16
    outputlayer3 = int(n_apis / 2) #8
    timesstep4 = int(n_apis / 4) # 4
    model = Sequential()
    model.add(LSTM(n_features, input_shape=(n_apis, train_X.shape[2]), return_sequences=True))  # train_X.shape[2] = 34
    model.add(LSTM(outputlayer2, return_sequences=True))
    model.add(LSTM(outputlayer3, return_sequences=True))
    model.add(LSTM(outputlayer2, return_sequences=True))
    model.add(LSTM(n_features, return_sequences=True))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_X, train_X, epochs=apa, batch_size=batch, shuffle=False)  # train
    return model
    #model.save(saved_model)


def train5modelAE(apa, batch, timestep, numfeature, filea, fileb, filec, filed, filee):
    #SingleFileLstmAutoencoder(apa, batch, n_apis, n_features, data_for_model_training)
    k1 = SingleFileLstmAutoencoder(apa, batch, timestep, numfeature, filea)
    k2 = SingleFileLstmAutoencoder(apa, batch, timestep, numfeature, fileb)
    k3 = SingleFileLstmAutoencoder(apa, batch, timestep, numfeature, filec)
    k4 = SingleFileLstmAutoencoder(apa, batch, timestep, numfeature, filed)
    k5 = SingleFileLstmAutoencoder(apa, batch, timestep, numfeature, filee)
    return k1, k2, k3, k4, k5


def postboosting(test_x, timestep, numfeature, apa, batch):
    yhatk1 = k1.predict(test_x)
    yhatk2 = k2.predict(test_x)
    yhatk3 = k3.predict(test_x)
    yhatk4 = k4.predict(test_x)
    yhatk5 = k5.predict(test_x)
    inputofdense = np.concatenate((yhatk1, yhatk2, yhatk3, yhatk4, yhatk5), 1).reshape(test_x.shape[0], 5, numfeature*timestep,1)
    test_x = test_x.reshape(test_x.shape[0], 1, numfeature*timestep, 1)
    modelmerge = Sequential()
    modelmerge.add(Conv2D(input_shape=(5, timestep*numfeature, 1), padding="valid", filters=1, kernel_size=(5, 1)))
    modelmerge.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')
    history = modelmerge.fit(inputofdense, test_x,validation_split=0.25,   nb_epoch=apa,  batch_size=batch)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    return modelmerge


#這邊是獨立建立五筆autoencoder

a = set()
while len(a)<5:
    a.add(random.choice(os.listdir(r"C:\Users\pups1\OneDrive\桌面\hooklogonehotforlstmautoencoder\hooklog")))#(r"/Users/yanyaosheng/Desktop/work_keras/final_csv/Browsefox/")))
    if len(a) == 5:
        break
print("using:")
print(list(a))
print("as five individual autoencoder.")
k1, k2, k3, k4, k5 = train5modelAE(apa, batch, timestep, numfeature, filenamelist+list(a)[0], filenamelist+list(a)[1], filenamelist+list(a)[2], filenamelist+list(a)[3], filenamelist+list(a)[4])

aaa = 0
for filename in listdir(filenamelist):
    test_x, scaler, y = data_preprocess(filenamelist + filename, timestep)
    test_x = test_x.reshape(test_x.shape[0], timestep, numfeature)
    if aaa == 0:
        new_test_x = test_x
        new_y = y
    else:
        new_test_x = np.concatenate([test_x, new_test_x], 0)
        new_y = np.concatenate([y, new_y], 0)
    aaa += 1
print
"total: " + str(aaa) + " files."

################################################################
# 上述把它資料夾內的檔案通通變成一個檔案格式(xxx, 16, 25)
################################################################


kkk = postboosting(new_test_x, timestep, numfeature, apa, batch)

################################################################
# kkk為訓練好的boosting檔案
################################################################
filepathlist = r"/Users/yanyaosheng/Desktop/work_keras/final_csv/Browsefox/"
a = []
for ii in range(aaa, (len(listdir(filenamelist)))):
    test_x, scaler, y = data_preprocess(filepathlist + listdir(filenamelist)[ii], timestep)
    test_x = test_x.reshape(test_x.shape[0], timestep, numfeature)
    yhatk1 = k1.predict(test_x)
    yhatk2 = k2.predict(test_x)
    yhatk3 = k3.predict(test_x)
    yhatk4 = k4.predict(test_x)
    yhatk5 = k5.predict(test_x)
    inputofdense = np.concatenate((yhatk1, yhatk2, yhatk3, yhatk4, yhatk5), 1).reshape(test_x.shape[0], 5,
                                                                                       timestep * numfeature, 1)
    yhat = kkk.predict(inputofdense)
    yhat = yhat.reshape(yhat.shape[0], timestep * numfeature)
    yhat = scaler.inverse_transform(yhat)
    print
    "==="
    rmse = np.sqrt(np.mean(((yhat - y) ** 2), axis=1))
    print
    listdir(filenamelist)[ii]
    print
    np.mean(rmse)
    a.append(listdir(filenamelist)[ii])
    a.append(np.mean(rmse))

thefile = open('test.txt', 'w')
for item in a:
    thefile.write("%s\n" % item)
thefile.close()
################################################################
# 上述是將單一家族資料夾內的所有資料都去跑boosting alg.並把rmse記錄起來 寫成檔案
################################################################