from pathlib import Path

import numpy as np
import sys

import buffer
import gui
import ml
import signals

#paths 
csvPath = "data/testData/nivel2.csv"
trainingDataDirPath = "data/train"
outputDirPath = "outputs/csv"
modelPath = "outputs/models/motorClassifier.pth"

#signal dicts
sensorColumns = signals.buildMotorColumns()
frequencyBands = [
    ("Energy1_A", 0, 49),
    ("Energy2_A", 50, 149),
    ("Energy3_A", 150, 300),
]

#buffering variables
sampleRate = 100
csvBufferSeconds = 1
liveBufferSeconds = 2
liveStepSeconds = 0.25


#serialCom variables
serialPort = "COM3"
baudRate = 115200
serialDelimiter = ","
serialTimeout = 1.0
serialEncoding = "utf-8"

# ML variables
trainingWindowSeconds = 1
trainingStepSeconds = 1
startRow = 0
tempWindowSeconds = 0.1
featureNames = ml.buildFeatureNames(sampleRate)
failureNames = ml.buildFailureNames()
failureIndexes = ml.buildFailureIndexes(failureNames)
hiddenSize = 32
epochCount = 200
learningRate = 0.001


# Builds one classifier instance from the configured feature and label sizes.
def buildClassifierModel():
    modelSizes = ml.buildModelSizes(featureNames, failureNames, hiddenSize)
    model = ml.buildModel(
        modelSizes["inputSize"],
        modelSizes["hiddenSize"],
        modelSizes["outputSize"],
    )
    return model


# Builds the label-to-CSV training map from the training directory layout.
def buildLabelledCsvPaths(trainingDataDirPath):
    trainingDataDir = Path(trainingDataDirPath)
    labelledCsvPaths = {}
    for failureName in failureNames:
        labelDir = trainingDataDir / failureName
        csvPaths = [
            str(csvFilePath)
            for csvFilePath in sorted(labelDir.glob("*.csv"))
        ]
        if len(csvPaths) > 0:
            labelledCsvPaths[failureName] = csvPaths
    return labelledCsvPaths


# Builds ML feature and optional inference output from the current signal data.
def buildMLData(signalData, modelPath):
    featureVector = ml.buildFeatureVector(signalData, featureNames)
    mlData = {
        "featureNames": featureNames,
        "featureVector": featureVector,
        "failureNames": failureNames,
        "failureIndexes": failureIndexes,
    }
    if Path(modelPath).exists():
        model = buildClassifierModel()
        model = ml.loadModel(model, modelPath)
        featureMatrix = np.expand_dims(featureVector, axis=0)
        featureTensor = ml.buildFeatureTensor(featureMatrix)
        probabilityTensor = ml.runInference(model, featureTensor)
        probabilityDict = ml.buildProbabilityDict(
            probabilityTensor,
            failureNames,
        )
        predictedLabel = ml.buildPredictedLabel(
            probabilityTensor,
            failureNames,
        )
        mlData["probabilityDict"] = probabilityDict
        mlData["predictedLabel"] = predictedLabel
    return mlData


# Orchestrates CSV input through buffer, signals, ML, then charting.
def runCSV(csvPath, modelPath):
    signalData = signals.readCSV(
        csvPath,
        sensorColumns,
        sampleRate,
        csvBufferSeconds,
        startRow,
        tempWindowSeconds,
        frequencyBands,
    )
    signalData["mlData"] = buildMLData(signalData, modelPath)
    gui.printCSVSummary(csvPath, sensorColumns, signalData)

    plotRawPath = str(
        Path(outputDirPath) / (Path(csvPath).stem + "_plotRaw.png")
    )
    gui.plotRaw(signalData, plotRawPath)
    print("plotRawPath:", plotRawPath)

    return signalData


# Orchestrates live input through buffer, signals, and ML.
def runLive(
    port,
    baudrate,
    delimiter,
    timeout,
    encoding,
    modelPath,
):
    for bufferData in buffer.liveRollingBuffer(
        port,
        sensorColumns,
        sampleRate,
        liveBufferSeconds,
        liveStepSeconds,
        baudrate,
        delimiter,
        timeout,
        encoding,
    ):
        signalData = signals.buildSignals(
            bufferData,
            sampleRate,
            tempWindowSeconds,
            frequencyBands,
        )
        signalData["mlData"] = buildMLData(signalData, modelPath)
        gui.printLiveSummary(sensorColumns, signalData)


# Orchestrates labelled CSV training, model fitting, and weight saving.
def runTraining(trainingDataDirPath, modelPath):
    labelledCsvPaths = buildLabelledCsvPaths(trainingDataDirPath)
    trainingSet = ml.buildTrainingSet(
        labelledCsvPaths,
        sensorColumns,
        sampleRate,
        trainingWindowSeconds,
        trainingStepSeconds,
        tempWindowSeconds,
        frequencyBands,
        featureNames,
        failureIndexes,
    )
    featureTensor = ml.buildFeatureTensor(trainingSet["featureMatrix"])
    labelTensor = ml.buildLabelTensor(trainingSet["labelVector"])
    model = buildClassifierModel()
    lossHistory = ml.trainModel(
        model,
        featureTensor,
        labelTensor,
        epochCount,
        learningRate,
    )
    Path(modelPath).parent.mkdir(parents=True, exist_ok=True)
    ml.saveModel(model, modelPath)

    probabilityTensor = ml.runInference(model, featureTensor[:1])
    probabilityDict = ml.buildProbabilityDict(probabilityTensor, failureNames)
    predictedLabel = ml.buildPredictedLabel(probabilityTensor, failureNames)
    trainingData = {
        "trainingSet": trainingSet,
        "lossHistory": lossHistory,
        "modelPath": modelPath,
        "probabilityDict": probabilityDict,
        "predictedLabel": predictedLabel,
    }
    gui.printTrainingSummary(trainingData)
    return trainingData


# Runs one pipeline mode from the current process arguments.
def main():
    source = sys.argv[1]
    if source == "csv":
        selectedCsvPath = csvPath
        if len(sys.argv) > 2:
            selectedCsvPath = sys.argv[2]
        pipelineOutput = runCSV(selectedCsvPath, modelPath)
    elif source == "train":
        selectedTrainingDataDirPath = trainingDataDirPath
        selectedModelPath = modelPath
        if len(sys.argv) > 2:
            selectedTrainingDataDirPath = sys.argv[2]
        if len(sys.argv) > 3:
            selectedModelPath = sys.argv[3]
        pipelineOutput = runTraining(
            selectedTrainingDataDirPath,
            selectedModelPath,
        )
    else:
        pipelineOutput = runLive(
            port=serialPort,
            baudrate=baudRate,
            delimiter=serialDelimiter,
            timeout=serialTimeout,
            encoding=serialEncoding,
            modelPath=modelPath,
        )

    print("pipelineOutputReady")
    return pipelineOutput


if __name__ == "__main__":
    main()
