import os
import traceback

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


fname = r"/Users/user/Desktop/hooklog/Bdmj/0e5d047639d4fa1284e222c534bb3998aebb0654d5464f39180c7ec3fdf3baff_3276.trace.hooklog"

k=0
j=0
filepathlist = gothrougheveryfile(dir)
listapi = []



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

    print(listapi)

    print(j)
    print(k)


apiset(filepathlist)


