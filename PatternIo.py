import sys
import os
import pickle
import numpy as np
from PIL import Image

# return 1 on success,
# 0 otherwise
PatDir = r'./patterns/CNN letter Dataset'
file1 = r'./Patfileall.pickle'
file2 = r'./Lblfileall.pickle'
file3 = r'./Patfiletrain.pickle'
file4 = r'./Lblfiletrain.pickle'
file5 = r'./Patfiletest.pickle'
file6 = r'./Lblfiletest.pickle'


def PreparePatterns(PatDir,  # patterns stored in sub-dirs, PatDir is the parent folder to all of the sub-dirs
                    PatFile,
                    # resulting pattern file in the binary format described in the document file including path
                    LblFile):  # resulting label file. also in the binary format as specified in the document file.
    characters = os.listdir(PatDir)
    # os.makedirs(r'./bytes')
    PF = []
    LF = []
    str1 = 'IMAG'
    str2 = 'LABL'
    N = len(characters)
    for i in range(N):
        PF.append(str1)
        LF.append(str2)
        character = characters[i]
        print(character, end="")
        if (ord(character) >= ord('A')) and (ord(character) <= ord('Z')):
            label = int(ord(character) - ord('A') + 10)
        else:
            label = int(character)
        character_path = os.path.join(PatDir, character)
        each = os.listdir(character_path)
        # if(i==0):print(each)
        N_each = len(each)  # 每个文件夹中的数目
        PF.append(N_each)
        LF.append(N_each)
        LF.append(label)
        m = int(100)
        n = int(75)
        # print(m.size)
        PF.append(m)
        PF.append(n)
        for j in range(N_each):
            # print("",end="")
            jpg = os.path.join(character_path, each[j])
            img = Image.open(jpg).convert('L')
            img = np.asarray(img)
            PF.append(img)

    # LF.append(str2)
    # LF.append(36)
    # for i in range(0, 37):
    #     LF.append(i)
    with open(PatFile, 'wb') as f:
        pickle.dump(PF, f)
    with open(LblFile, 'wb') as f:
        pickle.dump(LF, f)

    return 1


# return 1 on success,
# 0 otherwise
def SplitPatterns(PatFile,  # pattern file in the binary format described in the document
                  LblFile,  # label file.
                  TrainPatFile,  # where to put the patterns in training group
                  TrainLblFile,  # labels for the training patterns
                  TestPatFile,  # where to put the patterns in the test group
                  TestLblFile,  # labels for the test patterns
                  percent):  # how many patterns will be assigned to training group, in the format for example 0.70,
    # means 70 percent of the patterns will be assigned to training group
    with open(PatFile, 'rb') as f:
        PF = pickle.load(f, encoding='bytes')

    with open(LblFile, 'rb') as f:
        LF = pickle.load(f, encoding='bytes')
    str1 = 'IMAG'
    str2 = 'LABL'
    tnpf = []
    tnlf = []
    ttpf = []
    ttlf = []
    N1 = len(PF)
    N2 = len(LF)
    for i in range(N1):
        if (isinstance(PF[i], str) == True) and (PF[i] == str1):
            i += 1
            cnt = int(PF[i])
            cnt_split = int(cnt * percent)

            tnpf.append(str1)
            tnpf.append(cnt_split)

            ttpf.append(str1)
            ttpf.append(cnt - cnt_split)

            i += 1
            tnpf.append(PF[i])
            ttpf.append(PF[i])

            i += 1
            tnpf.append(PF[i])
            ttpf.append(PF[i])
            # i=0 ->i=1 [1:11] all | cs=7 [1:8] tn| [8:11] tt √
            i += 1
            tnpf.extend(PF[i:i + cnt_split])
            ttpf.extend(PF[i + cnt_split:i + cnt])
    for i in range(N2):
        if (isinstance(LF[i], str) == True) and (LF[i] == str2):
            i += 1
            cnt = int(LF[i])
            cnt_split = int(cnt * percent)
            tnlf.append(str2)
            tnlf.append(cnt_split)

            ttlf.append(str2)
            ttlf.append(cnt - cnt_split)

            i += 1
            tnlf.append(LF[i])
            ttlf.append(LF[i])
    with open(TrainPatFile, 'wb') as f:
        pickle.dump(tnpf, f)
    with open(TrainLblFile, 'wb') as f:
        pickle.dump(tnlf, f)
    with open(TestPatFile, 'wb') as f:
        pickle.dump(ttpf, f)
    with open(TestLblFile, 'wb') as f:
        pickle.dump(ttlf, f)
    return 1


def LoadPatterns(PatFile,  # pattern file in the binary format described in the word document.
                 LblFile):  # label file

    pats = []
    lbls = []
    with open(PatFile, 'rb') as f:
        PF = pickle.load(f, encoding='bytes')

    with open(LblFile, 'rb') as f:
        LF = pickle.load(f, encoding='bytes')
    N1 = len(PF)
    #print("len(PF)={}".format(N1))
    N2 = len(LF)
    str1 = 'IMAG'
    str2 = 'LABL'
    for i in range(N1):
        if (isinstance(PF[i], str) == True) and (PF[i] == str1):
            i += 1
            cnt = int(PF[i])
            #print("{},".format(cnt),end="")
            i += 3
            pats.extend(PF[i:i+cnt])

    for i in range(N2):
        if (isinstance(LF[i], str) == True) and (LF[i] == str2):
            i += 1
            cnt = int(LF[i])
            i += 1
            for j in range(cnt):
                lbls.append(LF[i])

    return pats, lbls  # this the read patterns, and lbls
    # patterns are in format [numbers][75*100] as a numpy array
    # labels are also in a 1-D numpy array


#PreparePatterns(PatDir, file1, file2)
#SplitPatterns(file1, file2, file3, file4, file5, file6, 0.7)
#A, B = LoadPatterns(file3, file4)
#print("\n{}".format(B))

