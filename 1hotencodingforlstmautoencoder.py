import os
import traceback
from pandas import read_csv
from pandas import DataFrame
import numpy as np



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


    #print(listarg)


    return listarg



def apiset(filepathlist):
    k = 0
    j = 0
    listapi = []
    for z in range(len(filepathlist)):

        #print(pathlist[j])


        with open(filepathlist[z]) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        encountertimestamp = False
        for i in range(len(content)):
            if encountertimestamp == True:
                #print(content[i])  # there will be apis
                #print(i)
                listapi.append(content[i])
                encountertimestamp = False
                k = k+1

            #print(i)

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


    print(j)
    print(k)

    listapi = set(listapi)
    listapi = list(listapi)

    #print(listapi)

    print(j)
    print(k)

    return listapi


apisetlist = apiset(filepathlist)
argumentsetlist = argumentset(filepathlist)

filenumber = 0
for filepath in filepathlist:
    print(filepath)
    with open(filepath) as f:
        content = f.readlines()
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
        print("len of apiset")
        print(len(apisetlist))
        print("len of argumentset")
        print(len(argumentsetlist))
        for j in range(len(listallapiintegerencode[i])):  # in one api
            if j == 0:  # time to encode api
                oneapi_onehotforapi[listallapiintegerencode[i][j]] = 1
            else:  # time to encode arg
                oneapi_onehotforarg[listallapiintegerencode[i][j]] = 1

        onehot = oneapi_onehotforapi + oneapi_onehotforarg  # onehot for one api
        onehotallapis.append(onehot)
        onehot = []


    onhotencode = np.array([np.array(x) for x in onehotallapis])  # nparray onehot encode for one hooklog
    df = DataFrame(onhotencode)
    desdir = "/Users/user/Desktop/hooklogonehotforlstmautoencoder2"
    df.to_csv("{0}/{1}.csv".format(desdir,os.path.basename(filepath)))

    filenumber= filenumber+1
    print("this is file number")
    print(filenumber)



#apiarglist = apisetlist+ argumentsetlist

#print(len(apiarglist))













