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
        for filename in os.listdir(dir+r"/"+subdir):
            if filename[0] == '.':
                continue
            wholefilepath = dir + r"/"+ subdir +r"/" +filename
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


classnamedir = [x[0] for x in os.walk(dir)]
classnamedir.remove(classnamedir[0])


# this method is for labeling
def whichclassitis(path):

    return classnamedir.index(os.path.dirname(path))



apisetlist = apiset(filepathlist)
argumentsetlist = argumentset(filepathlist)
allhooklog =[] # one element is a list that contains the info for one hooklog

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
    listallapiintegerencode = []  # one hooklog's apis integer encoding
    listoneapiandargcode = []  # for one api to be a list of integer inorder to onehot encoding
    for i in range(len(allapiandarglist)):  # i is index for apis
        for j in range(len(allapiandarglist[i])):  # j is index in one apis means apiname or its arg
            allapiandarglist[i][j]
            if j == 0:  # index 0 is always apiname
                indexapiname = apisetlist.index(allapiandarglist[i][j])
                listoneapiandargcode.append(indexapiname)
            else: # arg collection
                indexarg = argumentsetlist.index(allapiandarglist[i][j])
                listoneapiandargcode.append(indexarg)
        listallapiintegerencode.append(
            listoneapiandargcode)  # when one api encoded append its list to big list which included all apis
        listoneapiandargcode = []

    #print(listallapiintegerencode)

    onehot = [] # onehot is onehot encoding for one api in one hooklog
    onehotallapis = [] # all api means all apis in one hooklog is record in this two dim list

    # this version a onehot is a hooklog
    onehot = []

    onehotapi = [0] * len(apisetlist)
    onehotarg = [0] * len(argumentsetlist)

    for list_apiandarg in listallapiintegerencode:  # each api and arg (one row in csv)
        print(list_apiandarg[0])  # it is api
        onehotapi[list_apiandarg[0]] = 1  # time to encode api
        for arg in list_apiandarg[1:]:  # they are arg in one row
            print(arg)  # its arg
            onehotarg[arg] = 1  # time to encode arg


    #which class this sample belongs to
    index = whichclassitis(filepath)
    listinx = []
    listinx.append(index)
    # this section is the process of labeling for each sample(row in csv)

    onehot = onehotapi + onehotarg + listinx


    allhooklog.append(onehot)








#from two dim list to nparray

onhotencode = np.array([np.array(x) for x in allhooklog])  # nparray onehot encode for one hooklog
df = DataFrame(onhotencode)
df.to_csv("{0}.csv".format("onehotforclassifier"))



#apiarglist = apisetlist+ argumentsetlist

#print(len(apiarglist))













