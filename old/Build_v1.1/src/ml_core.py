################################################################################
# Imports                                                                      #
################################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

################################################################################
# variables/constants                                                          #
################################################################################
featureCols = [
    "acc1MagMean",
    "gyr1MagMean",
    "tempAvgMean",
    "tempGradMean",
    "ax1FundHz",
    "ax1FundMag",
    "ay1FundHz",
    "ay1FundMag",
    "az1FundHz",
    "az1FundMag",
    "acc2MagMean",
    "gyr2MagMean",
    "ax2FundHz",
    "ax2FundMag",
    "ay2FundHz",
    "ay2FundMag",
    "az2FundHz",
    "az2FundMag",
]

labelNames = [
    "healthy",
    "looseMounting",
    "maxLoading",
    "minLoading",
    "offAxis",
    "multipleFailures",
]
hiddenSize = 32

################################################################################
# helpers                                                                      #
################################################################################


# Converts one compact label name into a readable monitor label.
def displayLabel(labelName):
    labelChars = []
    prevAlphaNum = False
    for char in labelName:
        if char in ["_", "-"]:
            labelChars.append(" ")
            prevAlphaNum = False
            continue
        if char.isupper() and prevAlphaNum:
            labelChars.append(" ")
        labelChars.append(char)
        prevAlphaNum = char.islower() or char.isdigit()
    labelText = "".join(labelChars).strip().title()
    return labelText


# Encodes label names into integer indexes.
def encodeLabels(labelList):
    labelIndex = {}
    for index, labelName in enumerate(labelNames):
        labelIndex[labelName] = index
    labelVector = np.array(
        [labelIndex[labelName] for labelName in labelList],
        dtype=np.int64,
    )
    return labelVector


# Converts one feature matrix into a PyTorch float tensor.
def featureTensor(featureMatrix):
    tensorValue = torch.tensor(featureMatrix, dtype=torch.float32)
    return tensorValue


# Converts one label vector into a PyTorch long tensor.
def labelTensor(labelVector):
    tensorValue = torch.tensor(labelVector, dtype=torch.long)
    return tensorValue


# Builds the model size dictionary from the current feature and label lists.
def modelSizes(hiddenSize):
    sizeData = {
        "inputSize": len(featureCols),
        "hiddenSize": hiddenSize,
        "outputSize": len(labelNames),
    }
    return sizeData


################################################################################
# main functions                                                               #
################################################################################


# Builds one ordered feature vector from the shared signal dictionaries.
def featureVector(timeSignals, freqSignals):
    featDict = {
        "acc1MagMean": np.mean(timeSignals["acc1Mag"]),
        "gyr1MagMean": np.mean(timeSignals["gyr1Mag"]),
        "tempAvgMean": np.mean(timeSignals["tempAvg"]),
        "tempGradMean": np.mean(timeSignals["tempGrad"]),
        "ax1FundHz": freqSignals["acc1XFundHz"],
        "ax1FundMag": freqSignals["acc1XFundMag"],
        "ay1FundHz": freqSignals["acc1YFundHz"],
        "ay1FundMag": freqSignals["acc1YFundMag"],
        "az1FundHz": freqSignals["acc1ZFundHz"],
        "az1FundMag": freqSignals["acc1ZFundMag"],
        "acc2MagMean": np.mean(timeSignals["acc2Mag"]),
        "gyr2MagMean": np.mean(timeSignals["gyr2Mag"]),
        "ax2FundHz": freqSignals["acc2XFundHz"],
        "ax2FundMag": freqSignals["acc2XFundMag"],
        "ay2FundHz": freqSignals["acc2YFundHz"],
        "ay2FundMag": freqSignals["acc2YFundMag"],
        "az2FundHz": freqSignals["acc2ZFundHz"],
        "az2FundMag": freqSignals["acc2ZFundMag"],
    }
    featureArray = np.array(
        [featDict[featureCol] for featureCol in featureCols],
        dtype=np.float32,
    )
    return featureArray


# Defines the shared dense classifier used by training and live inference.
class MotorClassifier(nn.Module):
    # Builds the shared model layers.
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, outputSize),
        )

    # Runs one forward pass through the classifier.
    def forward(self, featureTensor):
        outputTensor = self.model(featureTensor)
        return outputTensor


# Builds one classifier instance from the current model sizes.
def model(hiddenSize):
    sizeData = modelSizes(hiddenSize)
    modelValue = MotorClassifier(
        sizeData["inputSize"],
        sizeData["hiddenSize"],
        sizeData["outputSize"],
    )
    return modelValue


# Trains one model from the current feature and label tensors.
def trainModel(modelValue, featureData, labelData, epochCount, learningRate):
    lossFn = nn.CrossEntropyLoss()
    optimiser = optim.Adam(modelValue.parameters(), lr=learningRate)
    lossHistory = []
    for _ in range(epochCount):
        optimiser.zero_grad()
        outputTensor = modelValue(featureData)
        lossValue = lossFn(outputTensor, labelData)
        lossValue.backward()
        optimiser.step()
        lossHistory.append(lossValue.item())
    return lossHistory


# Runs one inference pass and returns class probabilities.
def runModel(modelValue, featureData):
    outputTensor = modelValue(featureData)
    probTensor = torch.softmax(outputTensor, dim=1)
    return probTensor


# Builds the named probability dictionary from one inference row.
def probDict(probTensor):
    probRow = probTensor.detach().cpu().numpy()[0]
    probData = {}
    for index, labelName in enumerate(labelNames):
        probData[labelName] = float(probRow[index])
    return probData


# Returns the highest-probability label name.
def topLabel(probTensor):
    probRow = probTensor.detach().cpu().numpy()[0]
    topIndex = int(np.argmax(probRow))
    topName = labelNames[topIndex]
    return topName


# Saves the current model weights to disk.
def saveModel(modelValue, modelPath):
    torch.save(modelValue.state_dict(), modelPath)


# Loads model weights from disk into the current model instance.
def loadModel(modelValue, modelPath):
    modelValue.load_state_dict(torch.load(modelPath, map_location="cpu"))
    return modelValue
