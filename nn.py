import argparse
import sys
import os
import pickle
import numpy as np
import PatternIo
import neural
from PIL import Image
import argparse

# PatDir = r'./patterns/CNN letter Dataset'
# file1 = r'./Patfileall.pickle'
# file2 = r'./Lblfileall.pickle'
# file3 = r'./Patfiletrain.pickle'
# file4 = r'./Lblfiletrain.pickle'
# file5 = r'./Patfiletest.pickle'
# file6 = r'./Lblfiletest.pickle'
# mode = 'train'

parser = argparse.ArgumentParser(description='what')
parser.add_argument('--Patdir', default=r'./patterns/CNN letter Dataset', type=str, help='input files')
parser.add_argument('--file1', default=r'./Patfileall.pickle', type=str, help='path of bytes of all the Patfile')
parser.add_argument('--file2', default=r'./Lblfileall.pickle', type=str, help='path of bytes of all the Labelfile')
parser.add_argument('--file3', default=r'./Patfiletrain.pickle', type=str,
                    help='path of bytes of the Patfile for train')
parser.add_argument('--file4', default=r'./Lblfiletrain.pickle', type=str,
                    help='path of bytes of the Labelfile for train')
parser.add_argument('--file5', default=r'./Patfiletest.pickle', type=str,
                    help='path of bytes of the Patfile for test')
parser.add_argument('--file6', default=r'./Lblfiletest.pickle', type=str,
                    help='path of bytes of the Labelfile for test')
parser.add_argument('--mode', default='test', type=str,
                    help='train or test')
parser.add_argument('--netpath', default=r'./net1.pickle', type=str,
                    help='train for save or test for load')
args = parser.parse_args()
PatDir = args.Patdir
file1 = args.file1
file2 = args.file2
file3 = args.file3
file4 = args.file4
file5 = args.file5
file6 = args.file6
mode = args.mode
save_path = args.netpath
load_path = args.netpath
Net = neural.NeuralNetwork(7500, 128, 26, 0.1)  # 直接将每个像素点作为特征

# load date
PatternIo.SplitPatterns(file1, file2, file3, file4, file5, file6, 0.75)
train_data_temp, train_label_temp = PatternIo.LoadPatterns(file3, file4)
test_data_temp, test_label_temp = PatternIo.LoadPatterns(file5, file6)
if mode == 'train':
    print("mode is train")
    # trans train into one - hot
    N1 = len(train_data_temp)
    N2 = len(train_label_temp)
    train_data = []
    train_label = []
    for i in range(N1):
        if train_label_temp[i] >= 10:
            p = train_label_temp[i]
            p -= 10  # A的编码在这里转化为0 比较好处理了
            tmp = np.zeros(26)
            tmp[p] = 1  # into one hot
            # print(tmp)
            train_label.append(tmp)
            train_data.append(train_data_temp[i].flatten())
    epochs = 20
    Net.Train(feas=train_data, labels=train_label, epochs=epochs)

    Net.Save(save_path)
elif mode == 'test':
    print("mode is test")
    N1 = len(test_data_temp)
    N2 = len(test_label_temp)
    #print(test_label_temp)
    test_data = []
    test_label = []
    for i in range(N1):
        if test_label_temp[i] >= 10:
            p = test_label_temp[i]
            p -= 10
            tmp = np.zeros(26)
            tmp[p] = 1
            test_label.append(tmp)
            test_data.append(test_data_temp[i].flatten())

    #print(test_data)
    Net.Load(load_path)
    corrects, wrongs = Net.Evaluate(test_data, test_label)
    prec = corrects / (corrects + wrongs)
    print("precision: {} ".format(prec))
