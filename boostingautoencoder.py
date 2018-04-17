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


apa = 10
batch = 10
timestep = 8
numfeature =  6294
#filenamelist =  r"/Users/user/Desktop/hooklog/"

familydir = "/Users/user/Desktop/17family_trace_selected_param - with all dir/autoit"

def argumentset(filepathlist):
    listarg = []
    for z in range(len(filepathlist)): # z indicate which text

        with open(filepathlist[z]) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]

        for i in range(len(content)):
            if content[i].find("=")!=-1 or content[i].find(":")!=-1: #is a arg
                #print(content[i])
                #print(content[i].find("="))
                #print(content[i][u+1:])
                listarg.append(content[i])
    print(len(listarg))
    listarg = set(listarg)
    listarg = list(listarg)

    #print(listarg)
    print(len(listarg))

    return listarg



def apiset(filepathlist):
    k = 0
    j = 0
    listapi = []
    for z in range(len(filepathlist)):




        with open(filepathlist[z]) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        encountertimestamp = False
        for i in range(len(content)):
            if encountertimestamp == True:


                listapi.append(content[i])
                encountertimestamp = False
                k = k+1



            try:

                if content[i] != "":
                    if content[i][0] == '#':
                        encountertimestamp = True

                        j =j+1
            except :
                print("error"+str(i))
                print(len(content))
                print(content[i])
                traceback.print_exc()


        content =[]



    listapi = set(listapi)
    listapi = list(listapi)





    return listapi

def writefile(code,filepath):
    df = DataFrame(code)

    print("this is fucking shape")
    print(df.shape)
    desdir = "/Users/user/Desktop/hooklogonehotforlstmautoencoder3"
    print("this hello fucl ")
    print(os.path.dirname(filepath))
    print(os.path.exists(os.path.dirname(filepath)))


    if not os.path.isdir("{0}/{1}".format(desdir, filepath.split("//")[-2])):
        os.makedirs("{0}/{1}".format(desdir, filepath.split("//")[-2]))
    print("this is")

    df.to_csv("{0}/{1}/{2}.csv".format(desdir, filepath.split("//")[-2], os.path.basename(filepath)))

    print("go there11")








def featureencodeto10represent(filepath,filepathlist):
    filenumber = 0
    filepathlist = gothrougheveryfile(filepathlist)
    apisetlist = apiset(filepathlist)
    argumentsetlist = argumentset(filepathlist)
    print("this is apisetlist + argumentsetlist")
    print(len(apisetlist)+len(argumentsetlist))



    print(filepath)
    with open(filepath) as f:#open(filepathlist[299]) as f: #filepath[299]) as f:
        content = f.readlines()
        #print(content)
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    allapiandarglist = []
    apiinfolist = []
    encounterapi = False
    with open(filepath) as f:  # open(filepathlist[299]) as f: #filepath[299]) as f:
        content = f.readlines()
        # print(content)
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    allapiandarglist = []
    apiinfolist = []
    encounterapi = False
    for i in range(len(content)):
        try:
            if content[i] != "":

                if content[i][0] == "#":  # start to catch one api info

                    if len(apiinfolist) != 0:
                        allapiandarglist.append(apiinfolist)  # when encounter timestamp means go to another api

                    apiinfolist = []  # reinit everytime when encounter a timestamp
                    encounterapi = True
        except:
            print("i is number")
            print(i)
            print(len(content))
        if encounterapi == True:
            if content[i] != "":
                if content[i][0] != "#":
                    apiinfolist.append(content[i])


    #allapi = np.array([np.array(x) for x in allapiandarglist])  # convert two dim list to two dim array
    # allapi = np.array(allapiandarglist)

    # convert to integer format inspired by https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    listallapiintegerencode = []  # one hooklog's apis integer encoding . each element is a list which is a api and its arg integer encoding
    listoneapiandargcode = []  # for one api to be a list of integer inorder to onehot encoding, for example after running this for loop list will be [3,1800, 176x, ....]
    # which first element 3 means the integer encoding for api and 1800 and the element behind it are the integer encoding for argument
    for i in range(len(allapiandarglist)):  # i is index for apis

        for j in range(len(allapiandarglist[i])):  # j is index in one api means apiname or its arg
            allapiandarglist[i][j]
            if j == 0:  # index 0 is always apiname
                indexapiname = apisetlist.index(allapiandarglist[i][j])
                listoneapiandargcode.append(indexapiname)
            else: # arg collection
                indexarg = argumentsetlist.index(allapiandarglist[i][j])
                listoneapiandargcode.append(indexarg)
        listallapiintegerencode.append(listoneapiandargcode)  # when one api encoded append its list to big list which included all apis
        listoneapiandargcode = []

    #print(listallapiintegerencode)

    onehot = []
    onehotallapis = []

    for i in range(len(listallapiintegerencode)):  # for one api in one hooklog

        oneapi_onehotforapi = [0] * len(apisetlist)
        oneapi_onehotforarg = [0] * len(argumentsetlist)

        for j in range(len(listallapiintegerencode[i])):  # in one api
            #print("go there8")
            if j == 0:  # time to encode api
                oneapi_onehotforapi[listallapiintegerencode[i][j]] = 1
            else:  # time to encode arg
                oneapi_onehotforarg[listallapiintegerencode[i][j]] = 1

        onehot = oneapi_onehotforapi + oneapi_onehotforarg  # onehot for one api
        onehotallapis.append(onehot)
        onehot = []

    print("go there10")
    onehotencode = np.array([np.array(x) for x in onehotallapis])  # nparray onehot encode for one hooklog
    print(onehotencode.shape)
    print("go there9")
    #writefile(onehotencode,filepath)
    filenumber = filenumber + 1
    print("this is file number")
    print(filenumber)
    return onehotencode


def gothrougheveryfile(directory):
    pathlist = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".trace"):
                pathlist.append(os.path.join(root, file))
    print(len(pathlist))
    return pathlist


def data_to_reconstruction_problem(data, timestep):
    print("data_to_reconstruction_problem1")
    df = DataFrame(data)
    list_concat = list()
    for i in range(timestep - 1, -1, -1):
        tempdf = df.shift(i)
        list_concat.append(tempdf)
    data_for_autoencoder = concat(list_concat, axis=1)
    data_for_autoencoder.dropna(inplace=True)
    print("data_to_reconstruction_problem2")
    return data_for_autoencoder


def out_put_core(writting_list, outaddress, filename):
    thefile = open(outaddress + "\\" + filename + ".txt", 'w')
    for item in writting_list:
        thefile.write("%s\n" % item)


def data_preprocess(file_name, timestep):
    # read
    #dataset = read_csv(file_name,header = 0, index_col = 0) #header=None, index_col=None)
    encoding01 =featureencodeto10represent(file_name,familydir)
    if encoding01.shape[0]<16:
        return 0,0
    #values = encoding01.values
    reframed = data_to_reconstruction_problem(encoding01, timestep)


    values = reframed.values
    #reframed = reframed.astype('float32')
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled = scaler.fit_transform(reframed)
    #dfscaled = DataFrame(scaled)
    #valuescaled =dfscaled.values
    return  values,values


def SingleFileLstmAutoencoder(apa, batch, n_apis, n_features, data_for_model_training):
    #error_list = list()
    W_Hidden1_list = list()
    W_Hidden2_list = list()
    W_Hidden3_list = list()
    W_Hidden4_list = list()
    W_Hidden5_list = list()
    #現在改成讀進hooklog文字檔
    train_X, y = data_preprocess(data_for_model_training, n_apis)#146,400
    train_X = train_X.reshape((train_X.shape[0], n_apis, n_features))
    sample_number = train_X.shape[0]
    outputlayer2 = n_apis #16
    outputlayer3 = int(n_apis / 2) #8
    timesstep4 = int(n_apis / 4) # 4
    print("in SingleFileLstmAutoencoder1")
    model = Sequential()
    model.add(LSTM(n_features, input_shape=(n_apis, train_X.shape[2]), return_sequences=True))  # train_X.shape[2] = 34
    model.add(LSTM(outputlayer2, return_sequences=True))
    model.add(LSTM(outputlayer3, return_sequences=True))
    model.add(LSTM(outputlayer2, return_sequences=True))
    model.add(LSTM(n_features, return_sequences=True))
    print("in SingleFileLstmAutoencoder3")
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print("in SingleFileLstmAutoencoder4")
    history = model.fit(train_X, train_X, epochs=apa, batch_size=batch, shuffle=False)  # train
    print("in SingleFileLstmAutoencoder2")
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
familyfilelist =gothrougheveryfile(familydir)
while len(a)<5:

    a.add(random.choice(familyfilelist))#(r"/Users/yanyaosheng/Desktop/work_keras/final_csv/Browsefox/")))f
    if len(a) == 5:
        break

print("using:")
print(list(a))
print("as five individual autoencoder.")
k1, k2, k3, k4, k5 = train5modelAE(apa, batch, timestep, numfeature, list(a)[0], list(a)[1], list(a)[2], list(a)[3], list(a)[4])

aaa = 0
for filename in listdir(familydir):
    test_x, scaler, y = data_preprocess(familydir + filename, timestep)
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

a = []
for ii in range(aaa, (len(listdir(familydir)))):
    test_x, scaler, y = data_preprocess(familydir + listdir(familydir)[ii], timestep)
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
    listdir(familydir)[ii]
    print
    np.mean(rmse)
    a.append(listdir(familydir)[ii])
    a.append(np.mean(rmse))

thefile = open('test.txt', 'w')
for item in a:
    thefile.write("%s\n" % item)
thefile.close()
################################################################
# 上述是將單一家族資料夾內的所有資料都去跑boosting alg.並把rmse記錄起來 寫成檔案
################################################################