import os


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

listarg =[]



#with open(pathlist[z]) as f:
'''
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
    print(len(listarg))
    listarg = set(listarg)


    #print(listarg)
    print(len(listarg))

argumentset(filepathlist)
'''

def argumentset(filepathlist):
    listarg = []
    for z in range(len(filepathlist)): # z indicate which text

        with open(filepathlist[z]) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]

        for i in range(len(content)):
            if content[i].find("=")!=-1: #is a arg
                #print(content[i])
                #print(content[i].find("="))
                #print(content[i][u+1:])
                listarg.append(content[i])
    print(len(listarg))
    listarg = set(listarg)


    #print(listarg)
    print(len(listarg))

argumentset(filepathlist)


