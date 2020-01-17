import cv2 as cv
import numpy as np
import os
import torch
from classifier import CNN
import pyautogui as pg
from interaction import Mouse
import time

#Preparation
cap = cv.VideoCapture(0) #external camera
cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
cap.set(28, 0)
types = {0: 'Palm', 1: 'Hang', 2: 'Two', 3: 'Okay'}

#Sizes
dimensions = (3, 480,640)
screenDim = pg.size()
pg.position()
classes = len(types)
border = .2 #designates fraction of length or width for the border of detection region

#Taking pictures
pictureType = 3 #index the kind of gesture we are taking pictures of

#Saving and loading
savePath = 'dataTest/'
modelPath = ''
modelName = "deepDetector.txt"

#Mode
trainingMode = False
momentum = 5

class Detector():
    def __init__(self, dim):
        self.dimensions = dim
        self.backCaptured = False
        if not trainingMode:
            self.detector = CNN(classes, "")
            self.detector.load_state_dict(torch.load(modelPath + modelName))
            self.detector.eval()
            self.font = cv.FONT_HERSHEY_SIMPLEX
        self.back = None
        self.mouse = Mouse(screenDim)
        cut = border / 2
        Y = self.dimensions[1]
        X = self.dimensions[2]
        self.minX = cut * X
        self.minY = cut * Y
        self.maxX = X - self.minX
        self.maxY = Y - self.minY
        self.index = 0

        #Mouse
        self.velocity = 0
        self.prevAction = None
        self.clicked = False

    def removeBack(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(frame, 150, 255, cv.THRESH_OTSU)
        equal = cv.equalizeHist(frame)
        foreground = self.back.apply(frame, learningRate = 0)
        foreground = cv.medianBlur(foreground, 5)
        kernel = np.ones((4,4), np.uint8)
        dilated = cv.dilate(foreground, kernel, iterations = 2)
        erosion = cv.erode(dilated, kernel, iterations = 2)
        return erosion

    def handleAction(self, action, point):
        #if we have a different action and hasn't gained enough momentum
        if self.prevAction != action:
            if self.velocity < momentum:
                self.velocity += 1
                action = self.prevAction
            else:
                self.velocity = 0
                self.prevAction = action
        if action == 0:
            self.clicked = False
            self.mouse.releaseLeft()
            self.mouse.moveCursorTo(point[0], point[1])
        if action == 1:
            if not self.clicked:
                self.mouse.releaseLeft()
                self.mouse.singleClickLeft()
                self.clicked = True
        if action == 2:
            if not self.clicked:
                self.mouse.releaseLeft()
                self.mouse.doubleClickLeft()
                self.clicked = True
        if action == 3:
            self.clicked = False
            self.mouse.holdLeft()
            self.mouse.moveCursorTo(point[0], point[1])

    def extractPoint(self, hull):
        #Takes points of convex hull and converts to proportion of x and y respectively
        hull = [elem[0] for elem in hull]
        point = min(hull, key = lambda p: p[1]) #for now
        x = min(max(point[0], self.minX), self.maxX) - self.minX
        y = min(max(point[1], self.minY), self.maxY) - self.minY
        propX = x / (self.maxX - self.minX)
        propY = y / (self.maxY - self.minY)
        return (propX, propY)

    def execute(self):
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv.bilateralFilter(frame, 5, 50, 100)
            frame = cv.flip(frame, 1)
            if (ret is True) and (self.backCaptured):
                remove = self.removeBack(frame)
                contours, hierarchy = cv.findContours(remove, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                remove = cv.cvtColor(remove, cv.COLOR_GRAY2BGR)
                areas = [cv.contourArea(c) for c in contours]
                if len(areas) > 0:
                    maxA = max(areas)
                    if maxA > 2000 and maxA < 240000:
                        maxIdx = areas.index(maxA)
                        hull = cv.convexHull(contours[maxIdx])
                        cv.drawContours(frame, contours, maxIdx, (0,255,0), 3, 0)
                        cv.drawContours(frame, [hull], 0, (0, 0, 255), 3)
                        cv.drawContours(remove, contours, maxIdx, (0,255,0), 3, 0)
                        cv.drawContours(remove, [hull], 0, (0, 0, 255), 3)
                        if not trainingMode:
                            img = np.array(remove)
                            img = np.swapaxes(img, 0,2)
                            img = np.swapaxes(img, 1,2)
                            input = (torch.from_numpy(img).float()).view((1, self.dimensions[0], self.dimensions[1], self.dimensions[2]))
                            index = self.detector.predict(input)
                            if index != None:
                                proportions = self.extractPoint(hull)
                                action = types[index]
                                cv.putText(frame, action, (10, 50), self.font, 1, (255, 0, 0), 2, cv.LINE_AA)
                                cv.putText(remove, action, (10, 50), self.font, 1, (255, 0, 0), 2, cv.LINE_AA)
                                self.handleAction(index, proportions)
                cv.imshow('remove', remove)
            cv.imshow('frame', frame)
            key = cv.waitKey(1)
            if (key & 0xFF == ord('t')):
                self.back = cv.createBackgroundSubtractorMOG2(0, 50, detectShadows= False)
                self.backCaptured = True
            if (key & 0xFF == ord('p')):
                cv.imwrite(savePath + types[pictureType] + '_' + str(self.index) + ".jpg", remove)
                self.index += 1
            if (key & 0xFF == ord('q')) or (key & 0xFF == 27):
                break

d = Detector(dimensions)
d.execute()

cap.release()
cv.destroyAllWindows()
