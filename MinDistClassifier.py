import sys
import os
import pickle
import numpy as np
import PatternIo
from PIL import Image

PatDir = r'./patterns/CNN letter Dataset'
file1 = r'./Patfileall.pickle'
file2 = r'./Lblfileall.pickle'
file3 = r'./Patfiletrain.pickle'
file4 = r'./Lblfiletrain.pickle'
file5 = r'./Patfiletest.pickle'
file6 = r'./Lblfiletest.pickle'


def Get_each_mean(Patsfile, Lblsfile):
    means = []
    labels = []
    Pats, Lbls = PatternIo.LoadPatterns(Patsfile, Lblsfile)
    N1 = len(Pats)
    N2 = len(Lbls)
    # print(N1)
    # print(N2)
    if (N1 != N2):
        print("warining!patterns are not match Labels")
    str1 = 'IMAG'
    str2 = 'LABL'
    label1 = Lbls[0]
    total = Pats[0].copy()
    total = total.astype(float)
    # print(total+Pats[1])
    labels.append(Lbls[0])

    labelcount = 1
    flag = 0
    for i in range(1, N1):
        # if (isinstance(PF[i], str) == True) and (PF[i] == str1):
        if i >= 1 and Lbls[i] != Lbls[i - 1]:
            # print(Lbls[i])
            # if(flag==0):
            #     print(total)
            total = total / labelcount
            # if flag==0:
            #     print(total,Lbls[i])
            #     flag=1
            # print(labelcount)
            means.append(total)
            labelcount = 1
            labels.append(Lbls[i])
            total = Pats[i].copy()
            total = total.astype(float)
            continue
        elif i == N1 - 1:
            total += Pats[i]
            labelcount+=1
            total = total / labelcount
            means.append(total)
            break

        total += Pats[i]
        labelcount += 1
    return means, labels


def Predict_character(means, labels, jpg):  # jpg is image path
    N1 = len(means)
    (m, n) = means[0].shape
    # print(m,n)
    img = Image.open(jpg).convert('L')
    img = img.resize((n, m))
    img = np.asarray(img)
    # (m1,n1)=img.shape
    # print(m1,n1)

    N2 = len(labels)
    if (N1 != N2):
        print("warining!patterns are not match Labels")
    dis = []
    lbs = []  # lbls
    for i in range(N1):
        distance = np.linalg.norm((means[i] - img))
        dis.append(distance)
        lbs.append(labels[i])
    dis, lbs = zip(*sorted(zip(dis, lbs)))
    # list1, list2 = zip(*sorted(zip(list1, list2))
    ans = lbs[0]
    char1 = 'a'
    # ans2= lbs[0]
    # print(dis)
    if lbs[0] >= 10:
        char1 = chr(lbs[0] - 10 + ord('A'))
    else:
        char1 = ans
    return ans, char1


def Predict_character2(means, labels, img):  # img is numpy [75*100]
    N1 = len(means)
    (m, n) = means[0].shape
    # print(m,n)
    # img = Image.open(jpg).convert('L')
    # img = img.resize((n, m))
    # img = np.asarray(img)
    # (m1,n1)=img.shape
    # print(m1,n1)

    N2 = len(labels)
    if (N1 != N2):
        print("warining!patterns are not match Labels")
    dis = []
    lbs = []  # lbls
    for i in range(N1):
        distance = np.linalg.norm((means[i] - img))
        dis.append(distance)
        lbs.append(labels[i])
    dis, lbs = zip(*sorted(zip(dis, lbs)))
    # list1, list2 = zip(*sorted(zip(list1, list2))
    ans = lbs[0]
    char1 = 'a'
    # ans2= lbs[0]
    # print(dis)
    if lbs[0] >= 10:
        char1 = chr(lbs[0] - 10 + ord('A'))
    else:
        char1 = ans
    return ans, char1


def test_acc(means, lbls, testpat, testlbl):
    testpats, testlbls = PatternIo.LoadPatterns(testpat, testlbl)
    N1 = len(testpats)
    N2 = len(testlbls)
    if (N1 != N2):
        print("warining!patterns are not match Labels")
    acc = 0
    labelcount = 1
    for i in range(N1):
        if i >= 1 and testlbls[i] != testlbls[i - 1]:
            acc = acc / labelcount
            print("The number {} 's accuracy is ={}".format(testlbls[i - 1], acc))
            labelcount=1
            acc = 0
        elif i == N1 - 1:
            ans, char = Predict_character2(means, lbls, testpats[i])
            #print(testlbls[i])
            if (ans == testlbls[i]):
                acc += 1
            #print(acc)
            acc = acc / labelcount
            print("The number {} 's accuracy is ={}".format(testlbls[i], acc))
            continue
        ans, char = Predict_character2(means, lbls, testpats[i])
        if (ans == testlbls[i]):
            acc += 1
        labelcount += 1


PatternIo.SplitPatterns(file1, file2, file3, file4, file5, file6, 0.75)
A, B = Get_each_mean(file3, file4)
# test
imagepath = r'./aug8050_9.jpg'  # the image path
print("{} is predicted :  {}".format(imagepath, Predict_character(A, B, imagepath)[1]))
#test_acc(A, B, file5, file6)
