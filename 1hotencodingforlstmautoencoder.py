import os
import traceback
from pandas import read_csv
from pandas import DataFrame
import numpy as np
from pandas import concat




dir = "/Users/user/Desktop/hooklog"

def gothrougheveryfile(dir):  # go through every csv and concat to one csv this part has move to concatallcsv
    i = 0
    filenamelist = list()
    for subdir in os.listdir(dir):# outside each folder
        if subdir[0] == '.':
            continue
        for filename in os.listdir(dir+"//"+subdir):
            if filename[0] == '.':
                continue
            wholefilepath = dir + "//"+ subdir +"//" +filename
            filenamelist.append(wholefilepath)
    return filenamelist

filepathlist = gothrougheveryfile(dir)

def data_to_reconstruction_problem(data,timestep):
    df = DataFrame(data)
    list_concat=list()
    for i in range(timestep-1,-1,-1):
        tempdf=df.shift(i)
        list_concat.append(tempdf)
    data_for_autoencoder=concat(list_concat,axis=1)
    data_for_autoencoder.dropna(inplace=True)
    return data_for_autoencoder


def argumentset(filepathlist):
    listarg = []
    for z in range(len(filepathlist)): # z indicate which text

        with open(filepathlist[z]) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]

        for i in range(len(content)):
            if content[i].find("=")!=-1:
                #print(content[i])
                #print(content[i].find("="))
                u = content[i].find("=")
                #print(content[i][u+1:])
                listarg.append(content[i][u+1:])
            elif content[i].find(':') != -1:
                listarg.append(content[i])


    listarg = set(listarg)
    listarg = list(listarg)




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





apisetlist = apiset(filepathlist)
argumentsetlist = argumentset(filepathlist)

print("this is len of filepathlist")
print(len(filepathlist))
filenumber = 0
for filepath in filepathlist:
    print(filepath)
    with open(filepath) as f:#open(filepathlist[299]) as f: #filepath[299]) as f:
        content = f.readlines()
        #print(content)
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
                    if content[i].find("=") != -1:
                        u = content[i].find("=")
                        apiinfolist.append(content[i][u + 1:])

                    else:
                        apiinfolist.append(content[i])

    #allapi = np.array([np.array(x) for x in allapiandarglist])  # convert two dim list to two dim array
    # allapi = np.array(allapiandarglist)

    # convert to integer format inspired by https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    listallapiintegerencode = []  # one hooklog's apis integer encoding . each element is a list which is a api and its arg integer encoding
    listoneapiandargcode = []  # for one api to be a list of integer inorder to onehot encoding, for example after running this for loop list will be [3,1800, 176x, ....]
    # which first element 3 means the integer encoding for api and 1800 and the element behind it are the integer encoding for argument
    for i in range(len(allapiandarglist)):  # i is index for apis

        for j in range(len(allapiandarglist[i])):  # j is index in one apis means apiname or its arg
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
    writefile(onehotencode,filepath)
    filenumber = filenumber + 1
    print("this is file number")
    print(filenumber)





#apiarglist = apisetlist+ argumentsetlist

#print(len(apiarglist))













