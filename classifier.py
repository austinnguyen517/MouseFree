import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import random

'''
Define hyperparameters batchsize learning rate epochs split etc.

Define neural network architecture

Parse the jpg pictures and labels

Use train/test loader

Softmax function and negative log likelihood loss function

Plot testing and training loss
'''

#HYPERPARAMETERS
split = .8
batch = 16
lr = .0000001
epochs = 10
#NETWORK DEFINITION
path = 'dataTest/'
types = {0: 'Palm', 1: 'Hang', 2: 'Two', 3: 'Okay'}
indices = {'Palm': 0, 'Hang': 1, 'Two':2, 'Okay':3}
classes = len(types)
modelSaveName = "deepDetector.txt"

#ACTIONS:
trainingMode = False
testingMode = False

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DatasetImages(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

class CNN(nn.Module):
    def __init__(self, nOut, path):
        super(CNN, self).__init__()
        self.nOut = nOut
        self.loss_fnc = nn.CrossEntropyLoss()
        self.features = nn.Sequential(
            #extract features
            nn.Conv2d(3, 8, kernel_size = 3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(8, 16, kernel_size = 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(16, 32, kernel_size = 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(32, 64, kernel_size = 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(64, 128, kernel_size = 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(128, 512, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            #classify
            Flatten(),
            nn.Linear(9216, 4000),
            nn.ReLU(),
            nn.Linear(4000, 500),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, classes)
        )
        self.path = path

    def forward(self, input):
        out = self.features(input)
        return out

    def predict(self, input):
        self.eval()
        out = self.features(input)
        out = out.tolist()[0]
        #if self.softManual(out, out.index(max(out))) > .8:
        return out.index(max(out))
        #return None

    def softManual(self, output, index):
        #takes a list
        num = np.exp(output[index])
        denom = sum([np.exp(elem) for elem in output])
        return num/denom

    def getData(self, path):
        imageFiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        random.shuffle(imageFiles)
        random.shuffle(imageFiles)
        img = np.array(cv.imread(os.path.join(path, imageFiles[0])))
        img = np.swapaxes(img, 0,2)
        img = np.swapaxes(img, 1,2)
        dims = img.shape
        inputs = []
        outputs = []
        print("Iterating through files", flush = True)
        for file in imageFiles:
            img = np.array(cv.imread(os.path.join(path, file)))
            img = np.swapaxes(img, 0,2)
            img = np.swapaxes(img, 1,2)
            img = img.reshape((1, dims[0], dims[1], dims[2]))
            inputs.append(img)
            editted = file.replace("_", ".")
            editted = editted.split(".")
            index = indices[editted[0]]
            outputs.append(np.array([index]))

        print("Concatenating all images", flush = True)
        inputs = torch.FloatTensor(np.concatenate(inputs, axis = 0))
        outputs = torch.FloatTensor(np.concatenate(outputs, axis=0))

        trainInputs = inputs.narrow(0, 0, int(split*len(imageFiles)))
        trainOutputs = outputs.narrow(0,0,int(split*len(imageFiles)))

        testInputs = inputs.narrow(0, int(split*len(imageFiles)), len(imageFiles) - int(split*len(imageFiles)))
        testOutputs = outputs.narrow(0, int(split*len(imageFiles)), len(imageFiles) - int(split*len(imageFiles)))

        trainDataset = DatasetImages(trainInputs, trainOutputs)
        testDataset = DatasetImages(testInputs, testOutputs)
        return trainDataset, testDataset

    def train_cust(self):
        path = self.path
        print("Gathering and parsing data...", flush = True)
        train, test = self.getData(path)
        print("Finished parsing data", flush = True)
        print("")
        trainLoss, testLoss = self.optimize(train, test)
        return trainLoss, testLoss

    def test_cust(self):
        path = self.path
        train, test = self.getData(path)
        loader = DataLoader(train, batch_size = batch, shuffle = True)
        self.eval()
        stats = {}
        correct = 0
        counts = [0 for i in range(len(types))]
        avgLoss = 0
        for i, (input, target) in enumerate(loader):
            print("Progress: ", (i / len(loader)) * 100, " percent", flush = True)
            output = self.forward(input)
            loss = ((self.loss_fnc(output, target.long())).detach()).item()
            avgLoss += loss / len(loader)
            out = output.tolist()[0]
            counts[int(target.item())] += 1
            #Get probability of the item. Softmax manual
            prob = self.softManual(out, int(target.item()))

            curr = stats.get(types[int(target.item())], 0)
            curr += prob
            stats[types[target.item()]] = curr

            #Get correctness ratio
            j = out.index(max(out))
            if j == target.item():
                correct += 1
        for key in stats.keys():
            stats[key] = stats[key] / counts[indices[key]]
        return stats, (correct / len(loader)), avgLoss

    def optimize(self, train, test):
        #make a train and test loader. pass into network. So call log likelihood loss. optimizer with Adam
        trainLoader = DataLoader(train, batch_size = batch, shuffle = True)
        testLoader = DataLoader(test, batch_size = batch, shuffle = True)
        optimizer = torch.optim.Adam(super(CNN, self).parameters(), lr = lr)
        train = []
        test = []
        print("Beginning training", flush = True)
        for epoch in range(epochs):
            print("Epoch: ", epoch + 1, flush = True)
            numBatches = len(trainLoader)/batch
            train_error = torch.zeros(1)
            self.train()
            for i, (input, target) in enumerate(trainLoader):
                optimizer.zero_grad()
                output = self.forward(input)
                target = target.long()
                loss = self.loss_fnc(output, target)
                if loss != loss or (any(torch.isnan(val).byte().any() for val in self.state_dict().values())):
                    print('Nan values detected. Exiting.')
                    break
                loss.backward()
                optimizer.step()
                train_error += loss.item()
            train.append(train_error.item()/len(trainLoader))
            self.eval()
            test_error = torch.zeros(1)
            torch.save(network.state_dict(), modelSaveName)
            for i, (input, target) in enumerate(testLoader):
                output = self.forward(input)
                target = target.long()
                loss = self.loss_fnc(output, target)
                test_error += loss.item()
            test.append(test_error.item()/len(testLoader))
            print("Train error: ", train[-1], "     Test Error: ", test[-1])
            print("Model saved", flush = True)
        return train, test

if trainingMode:
    network = CNN(classes, path)
    network.load_state_dict(torch.load("deepDetector.txt"))
    trainError, testError = network.train_cust()
    plt.plot(trainError, label = "training")
    plt.plot(testError, label = "testing")
    plt.legend()
    plt.show()
if testingMode:
    network = CNN(classes, path)
    split = 1
    batch = 1
    network.load_state_dict(torch.load("deepDetector.txt"))
    stats, correct, avgLoss = network.test_cust()
    for stat in stats.items():
        print(stat[0], " with avg probability ", stat[1])
    print("Accuracy: ", correct)
    print("Average Loss: ", avgLoss)
