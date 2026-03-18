import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import buffer
import signals


# Returns the ordered feature names used by the ML model.
def featureNames(sampleRate):
    featureNames = [
        "mpu1AccMagMean",
        "mpu1GyrMagMean",
        "mpu1TempAvgMean",
        "mpu1TempGradMean",
        "ds18b20OneAvgMean",
        "ds18b20OneGradMean",
        "mpu1AccXFundamentalHz",
        "mpu1AccXFundamentalMag",
        "mpu1AccYFundamentalHz",
        "mpu1AccYFundamentalMag",
        "mpu1AccZFundamentalHz",
        "mpu1AccZFundamentalMag",
        "mpu2AccMagMean",
        "mpu2GyrMagMean",
        "mpu2TempAvgMean",
        "mpu2TempGradMean",
        "ds18b20TwoAvgMean",
        "ds18b20TwoGradMean",
        "mpu2AccXFundamentalHz",
        "mpu2AccXFundamentalMag",
        "mpu2AccYFundamentalHz",
        "mpu2AccYFundamentalMag",
        "mpu2AccZFundamentalHz",
        "mpu2AccZFundamentalMag",
    ]
    return featureNames


# Returns the ordered failure-state names used by the classifier.
def failureNames():
    failureNames = [
        "healthy",
        "looseMounting",
        "maxLoading",
        "minLoading",
        "offAxis",
        "multipleFailures",
    ]
    return failureNames


# Returns the failure-state descriptions keyed by state name.
def failureDescriptions():
    failureDescriptions = {
        "healthy": "Normal operation baseline block.",
        "looseMounting": "Mounting screws or bolts have loosened.",
        "maxLoading": "Motor load is at a sustained high-load state.",
        "minLoading": "Motor load is at a sustained low-load state.",
        "offAxis": "Motor axis is tilted relative to the mount.",
        "multipleFailures": "More than one induced failure state is present.",
    }
    return failureDescriptions


# Returns the integer label index for each failure-state name.
def failureIndexes(failureNames):
    failureIndexes = {}
    for index, failureName in enumerate(failureNames):
        failureIndexes[failureName] = index
    return failureIndexes


# Builds one named feature dictionary from the current signal data.
def featureDict(signalData):
    timeSignals = signalData["timeSignals"]
    freqSignals = signalData["freqSignals"]
    featureDict = {
        "mpu1AccMagMean": np.mean(timeSignals["mpu1AccMag"]),
        "mpu1GyrMagMean": np.mean(timeSignals["mpu1GyrMag"]),
        "mpu1TempAvgMean": np.mean(timeSignals["mpu1TempAvg"]),
        "mpu1TempGradMean": np.mean(timeSignals["mpu1TempGrad"]),
        "ds18b20OneAvgMean": np.mean(timeSignals["ds18b20OneAvg"]),
        "ds18b20OneGradMean": np.mean(timeSignals["ds18b20OneGrad"]),
        "mpu1AccXFundamentalHz": (
            freqSignals["mpu1AccXFundamentalHz"]
        ),
        "mpu1AccXFundamentalMag": (
            freqSignals["mpu1AccXFundamentalMag"]
        ),
        "mpu1AccYFundamentalHz": (
            freqSignals["mpu1AccYFundamentalHz"]
        ),
        "mpu1AccYFundamentalMag": (
            freqSignals["mpu1AccYFundamentalMag"]
        ),
        "mpu1AccZFundamentalHz": (
            freqSignals["mpu1AccZFundamentalHz"]
        ),
        "mpu1AccZFundamentalMag": (
            freqSignals["mpu1AccZFundamentalMag"]
        ),
        "mpu2AccMagMean": np.mean(timeSignals["mpu2AccMag"]),
        "mpu2GyrMagMean": np.mean(timeSignals["mpu2GyrMag"]),
        "mpu2TempAvgMean": np.mean(timeSignals["mpu2TempAvg"]),
        "mpu2TempGradMean": np.mean(timeSignals["mpu2TempGrad"]),
        "ds18b20TwoAvgMean": np.mean(timeSignals["ds18b20TwoAvg"]),
        "ds18b20TwoGradMean": np.mean(timeSignals["ds18b20TwoGrad"]),
        "mpu2AccXFundamentalHz": (
            freqSignals["mpu2AccXFundamentalHz"]
        ),
        "mpu2AccXFundamentalMag": (
            freqSignals["mpu2AccXFundamentalMag"]
        ),
        "mpu2AccYFundamentalHz": (
            freqSignals["mpu2AccYFundamentalHz"]
        ),
        "mpu2AccYFundamentalMag": (
            freqSignals["mpu2AccYFundamentalMag"]
        ),
        "mpu2AccZFundamentalHz": (
            freqSignals["mpu2AccZFundamentalHz"]
        ),
        "mpu2AccZFundamentalMag": (
            freqSignals["mpu2AccZFundamentalMag"]
        ),
    }
    return featureDict


# Builds one ordered NumPy feature vector from the current signal data.
def featureVector(signalData, featureNames):
    featDict = featureDict(signalData)
    featureVector = np.array(
        [featDict[featureName] for featureName in featureNames],
        dtype=np.float32,
    )
    return featureVector


# Builds valid start rows for repeated fixed-size training windows.
def windowStartRows(rowCount, sampleRate, bufferSeconds, stepSeconds):
    windowSamples = int(sampleRate * bufferSeconds)
    stepSamples = int(sampleRate * stepSeconds)
    lastStartRow = rowCount - windowSamples
    startRows = []
    for startRow in range(0, lastStartRow + 1, stepSamples):
        startRows.append(startRow)
    return startRows


# Builds one signal window from a CSV DataFrame slice.
def windowSignalData(
    dataFrame,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    periodSeconds,
    startRow,
    tempWindowSeconds,
    fftConfig,
):
    bufferData = buffer.buildBuf(
        dataFrame,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        startRow,
    )
    signalData = signals.buildSigs(
        bufferData,
        sampleRate,
        periodSeconds,
        tempWindowSeconds,
        fftConfig,
    )
    return signalData


# Builds labelled feature rows from repeated windows of one CSV file.
def labelledFeatureRows(
    csvPath,
    labelName,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    periodSeconds,
    stepSeconds,
    tempWindowSeconds,
    fftConfig,
    featureNames,
    failureIndexes,
):
    dataFrame = pd.read_csv(csvPath)
    startRows = windowStartRows(
        dataFrame.shape[0],
        sampleRate,
        bufferSeconds,
        stepSeconds,
    )
    featureRows = []
    labelNames = []
    for startRow in startRows:
        signalData = windowSignalData(
            dataFrame,
            sensorColumns,
            sampleRate,
            bufferSeconds,
            periodSeconds,
            startRow,
            tempWindowSeconds,
            fftConfig,
        )
        featVec = featureVector(signalData, featureNames)
        featureRows.append(featVec)
        labelNames.append(labelName)
    featureMatrix = np.vstack(featureRows)
    labelVector = encodeLabels(labelNames, failureIndexes)
    labelledRows = {
        "featureMatrix": featureMatrix,
        "labelVector": labelVector,
        "labelNames": labelNames,
        "startRows": startRows,
    }
    return labelledRows


# Builds one combined training set from multiple labelled CSV files.
def trainingSet(
    labelledCsvPaths,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    periodSeconds,
    stepSeconds,
    tempWindowSeconds,
    fftConfig,
    featureNames,
    failureIndexes,
):
    featureMatrices = []
    labelVectors = []
    datasetMeta = []
    for labelName, csvPaths in labelledCsvPaths.items():
        for csvPath in csvPaths:
            labelledRows = labelledFeatureRows(
                csvPath,
                labelName,
                sensorColumns,
                sampleRate,
                bufferSeconds,
                periodSeconds,
                stepSeconds,
                tempWindowSeconds,
                fftConfig,
                featureNames,
                failureIndexes,
            )
            featureMatrices.append(labelledRows["featureMatrix"])
            labelVectors.append(labelledRows["labelVector"])
            datasetMeta.append(
                {
                    "labelName": labelName,
                    "csvPath": csvPath,
                    "windowCount": len(labelledRows["startRows"]),
                }
            )
    trainingSet = {
        "featureMatrix": np.vstack(featureMatrices),
        "labelVector": np.concatenate(labelVectors),
        "datasetMeta": datasetMeta,
    }
    return trainingSet


# Encodes failure-state names into integer labels.
def encodeLabels(labelNames, failureIndexes):
    labelVector = np.array(
        [failureIndexes[labelName] for labelName in labelNames],
        dtype=np.int64,
    )
    return labelVector


# Converts feature vectors into a PyTorch float tensor.
def featureTensor(featureMatrix):
    featureTensor = torch.tensor(featureMatrix, dtype=torch.float32)
    return featureTensor


# Converts integer labels into a PyTorch long tensor.
def labelTensor(labelVector):
    labelTensor = torch.tensor(labelVector, dtype=torch.long)
    return labelTensor


# Defines a minimal dense classifier for block-based motor features.
class MotorClassifier(nn.Module):
    # Builds the model layers for motor-state classification.
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


# Builds the motor classifier instance.
def model(inputSize, hiddenSize, outputSize):
    model = MotorClassifier(inputSize, hiddenSize, outputSize)
    return model


# Builds the default model sizes from the configured feature and label sets.
def modelSizes(featureNames, failureNames, hiddenSize):
    modelSizes = {
        "inputSize": len(featureNames),
        "hiddenSize": hiddenSize,
        "outputSize": len(failureNames),
    }
    return modelSizes


# Trains the model on labelled feature tensors.
def trainModel(model, featureTensor, labelTensor, epochCount, learningRate):
    lossFunction = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learningRate)
    lossHistory = []
    for _ in range(epochCount):
        optimiser.zero_grad()
        outputTensor = model(featureTensor)
        loss = lossFunction(outputTensor, labelTensor)
        loss.backward()
        optimiser.step()
        lossHistory.append(loss.item())
    return lossHistory


# Runs inference and returns class probabilities for one feature tensor.
def runInference(model, featureTensor):
    outputTensor = model(featureTensor)
    probabilityTensor = torch.softmax(outputTensor, dim=1)
    return probabilityTensor


# Builds a named probability dictionary from one inference output row.
def probabilityDict(probabilityTensor, failureNames):
    probabilityRow = probabilityTensor.detach().cpu().numpy()[0]
    probabilityDict = {}
    for index, failureName in enumerate(failureNames):
        probabilityDict[failureName] = float(probabilityRow[index])
    return probabilityDict


# Returns the highest-probability failure-state name.
def predictedLabel(probabilityTensor, failureNames):
    probabilityRow = probabilityTensor.detach().cpu().numpy()[0]
    predictedIndex = int(np.argmax(probabilityRow))
    predictedLabel = failureNames[predictedIndex]
    return predictedLabel


# Saves model weights to disk.
def saveModel(model, modelPath):
    torch.save(model.state_dict(), modelPath)


# Loads model weights from disk into a model instance.
def loadModel(model, modelPath):
    model.load_state_dict(torch.load(modelPath, map_location="cpu"))
    return model
