import sys
import numpy as np
import time
import pickle  # used to load and store trained nn


def sigmoid(x):
    x_ravel = x.ravel()  # 将numpy数组展平
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)


class NeuralNetwork:
    def __init__(self,
                 NumInput,
                 NumHidden,
                 NumOutput,
                 LearningRate):
        self.input = NumInput
        self.hidden = NumHidden
        self.output = NumOutput
        self.LearningRate = LearningRate

        # input to hidden layer 
        # complete this part 
        w = np.random.randn(self.hidden, self.input)
        w /= np.sqrt(self.input)
        self.w1 = w

        w = np.random.randn(self.output, self.hidden)
        w /= np.sqrt(self.hidden)
        self.w2 = w

        # end of complete this part

    def TrainATime(self,  # reference to us
                   x,  # is one input vector
                   y):  # it label for this input vector, in one-hot format
        # forward pass
        # input to hidden
        # print("train a time")
        x = np.array(x, ndmin=2).T  # convert to 2d array ...其实是转置..行向量变列向量
        I1 = np.dot(self.w1, x)  # 矩阵乘法
        # print( "I1=", I1 )

        O1 = sigmoid(I1)

        # hidden to output
        I2 = np.dot(self.w2, O1)
        O = sigmoid(I2)

        # backward propagation
        y = np.array(y, ndmin=2).T  # convert to 2d array
        # print(y.shape)
        delta = (y - O) * O * (1. - O)

        dw = self.LearningRate * np.dot(delta, O1.T)

        self.w2 += dw

        t = np.dot(self.w2.T, delta)
        delta = t * O1 * (1 - O1)
        dw = self.LearningRate * np.dot(delta, x.T)

        self.w1 += dw

        return

    def RunATime(self,  # reference to us
                 x):  # is one input vector
        O = None  # to supress syntax error, O is initialzed with None
        # forward pass
        # input to hidden
        x = np.array(x, ndmin=2).T  # convert to 2d array
        I1 = np.dot(self.w1, x)
        # print( "I1=", I1 )

        O1 = sigmoid(I1)
        #print(O1)
        # hidden to output
        I2 = np.dot(self.w2, O1)
        O = sigmoid(I2)
        #print(O)
        return O

        # return the output for this input
        # return O

    def Evaluate(self,  # reference to us
                 feas,  # input feature vectors, can be many
                 labels):  # input labels for the feature vectors, in one-hot
        corrects, wrongs = 0, 0
        n = len(feas)
        for i in range(n):
            o = NeuralNetwork.RunATime(self, feas[i])
            r = o.argmax()
            c = labels[i].argmax()
            #print(r)
            if r == c:
                corrects += 1
            else:
                wrongs += 1

        # return corrects, wrongs
        return corrects, wrongs  # corrects contain number of correctly classified feas,
        # wrongs, number of errorly classified.

    # training interface
    def Train(self,
              feas,  # feature vectors used for training
              labels,  # corresponding labels for the feature vectors
              epochs):  # how many epochs to train, an epoch means training the nn
        # with all feature vectors once.

        # !call TrainATime to train
        # !TrainATime( ------)
        for i in range(epochs):
            start = time.time()
            n = len(feas)
            print("epoch={},n={},".format(i, n), end='')
            for j in range(n):  # number of samples
                x = feas[j]  # 64 element feature vector
                y = labels[j]
                NeuralNetwork.TrainATime(self, x, y)
            end = time.time()
            print("spend time = {}s".format(end - start))

            corrects, wrongs = NeuralNetwork.Evaluate(self, feas, labels)
            prec = corrects / (corrects + wrongs)
            print("epoch:{} precision: {} ".format(i,prec))

        # !call Eavluate to see how well we are now at
        # !corrects, wrongs = Evaluate( nn, feas, labels )
        # prec = corrects / (corrects + wrongs)
        # print("epoch: ", i, "precision: %.5f" % prec)

    # inference interface
    def Run(self,  # reference to us
            feas):  # feature vectors
        n = len(feas)
        res = []
        for i in range(n):
            x = feas[i]
            label = NeuralNetwork.RunATime(x)
            res.append(label.argmax())
        # will return the classification results in res
        return res

    # store the trained nn to a specified file, using pickle
    # on success, return 1
    # 0 otherwise.
    def Save(self,
             obj_file):  # file name including path #obj_file=../../**.pickle 或许也可以直接保存w1,w2
        try:
            with open(obj_file, 'wb') as f:
                ws=[]
                ws.append(self.w1)
                ws.append(self.w2)
                pickle.dump(ws, f)
            return 1
        except:
            return 0

    # load nn from file
    # on success, return 1
    # 0 otherwise.
    def Load(self,
             obj_file):
        # try:
        #     with open(obj_file, 'rb') as f:
        #         ws=pickle.load(obj_file)
        #         [self.w1,self.w2]=ws
        #     return 1
        # except:
        #     print("Load_False")
        #     return 0
        with open(obj_file, 'rb') as f:
            ws = pickle.load(f, encoding='bytes')
            self.w1=ws[0]
            self.w2=ws[1]
        return 1

# networksavepath = r'./networks.pickle'
# a = NeuralNetwork(64, 25, 10, 0.01)
# print(a.w1)
# a.Save(networksavepath)
# b = NeuralNetwork(64, 25, 10, 0.01)
# b.Load(networksavepath)
# print("b.w1=..")
# print(b.w1)
