from pathlib import Path

import numpy as np
import pandas as pd
import sys

import buffer
import gui
import ml
import signals

#paths
csvPath = "data/testData/test.csv"
trainingDataDirPath = "data/train"
outputDirPath = "outputs/csvCharts"
modelPath = "outputs/models/motorClassifier.pth"

#buffering variables
sampleRate = 1000
bufferSeconds = 2
periodSeconds = 1
csvPlotPeriodCount = 3
longPlotSeconds = 60
liveStepSeconds = periodSeconds

#signal dicts
sensorColumns = buffer.motorCols()
fftConfig = {
    "minFundHz": 1.0,
    "maxFundHz": 60.0,
    "plotMaxHz": 500.0,
    "bpfoLowOrder": 3.0,
    "bpfoHighOrder": 5.0,
    "bpfiLowOrder": 5.0,
    "bpfiHighOrder": 7.0,
    "orderToleranceFraction": 0.10,
}


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
tempWindowSeconds = 0.01
featureNames = ml.featureNames(sampleRate)
failureNames = ml.failureNames()
failureIndexes = ml.failureIndexes(failureNames)
hiddenSize = 32
epochCount = 200
learningRate = 0.001


# Builds one classifier instance from the configured feature and label sizes.
def clfModel():
    modelSizes = ml.modelSizes(featureNames, failureNames, hiddenSize)
    model = ml.model(
        modelSizes["inputSize"],
        modelSizes["hiddenSize"],
        modelSizes["outputSize"],
    )
    return model


# Builds the label-to-CSV training map from the training directory layout.
def labelCsvPaths(trainingDataDirPath):
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
def mlData(signalData, modelPath):
    featureVector = ml.featureVector(signalData, featureNames)
    mlData = {
        "featureNames": featureNames,
        "featureVector": featureVector,
        "failureNames": failureNames,
        "failureIndexes": failureIndexes,
    }
    if Path(modelPath).exists():
        model = clfModel()
        model = ml.loadModel(model, modelPath)
        featureMatrix = np.expand_dims(featureVector, axis=0)
        featureTensor = ml.featureTensor(featureMatrix)
        probabilityTensor = ml.runInference(model, featureTensor)
        probabilityDict = ml.probabilityDict(
            probabilityTensor,
            failureNames,
        )
        predictedLabel = ml.predictedLabel(
            probabilityTensor,
            failureNames,
        )
        mlData["probabilityDict"] = probabilityDict
        mlData["predictedLabel"] = predictedLabel
    return mlData


# Builds three spaced one-second CSV plot start rows for full buffers.
def csvPlotStartRows(
    rowCount,
    sampleRate,
    bufferSeconds,
    periodSeconds,
    periodCount,
):
    bufferSamples = int(sampleRate * bufferSeconds)
    stepSamples = int(sampleRate * periodSeconds)
    maxStartRow = rowCount - bufferSamples
    if maxStartRow < 0:
        return [0]
    startRows = []
    for startRow in range(0, maxStartRow + 1, stepSamples):
        startRows.append(startRow)
    targetPeriodCount = min(periodCount, len(startRows))
    if targetPeriodCount <= 1:
        return [0]
    rowIndexes = np.linspace(0, len(startRows) - 1, num=targetPeriodCount)
    selectedStartRows = []
    for rowIndex in rowIndexes:
        startRow = startRows[int(round(rowIndex))]
        if startRow not in selectedStartRows:
            selectedStartRows.append(startRow)
    return selectedStartRows


# Builds the short chart file names for one plotted CSV period.
def csvPlotPaths(outputDirPath, periodNumber):
    plotPaths = {
        "sensor1": str(
            Path(outputDirPath) / "sensors1" / f"s1p{periodNumber}.png"
        ),
        "sensor2": str(
            Path(outputDirPath) / "sensors2" / f"s2p{periodNumber}.png"
        ),
        "fftGrid": str(
            Path(outputDirPath) / "allFFT" / f"fftp{periodNumber}.png"
        ),
    }
    return plotPaths


# Builds the long chart file names in the dedicated long plot directory.
def csvLongPlotPaths(outputDirPath):
    plotPaths = {
        "sensor1": str(Path(outputDirPath) / "longs" / "s1l.png"),
        "sensor2": str(Path(outputDirPath) / "longs" / "s2l.png"),
        "fftGrid": str(Path(outputDirPath) / "longs" / "fftl.png"),
    }
    return plotPaths


# Creates the chart output directories used by CSV plotting.
def csvChartDirs(outputDirPath):
    outputDirs = [
        Path(outputDirPath),
        Path(outputDirPath) / "sensors1",
        Path(outputDirPath) / "sensors2",
        Path(outputDirPath) / "allFFT",
        Path(outputDirPath) / "longs",
    ]
    for outputDir in outputDirs:
        outputDir.mkdir(parents=True, exist_ok=True)


# Builds one long-duration CSV signal block for exploratory plotting.
def longSignalData(dataFrame, sampleRate, longPlotSeconds, tempWindowSeconds):
    bufferData = buffer.buildBuf(
        dataFrame,
        sensorColumns,
        sampleRate,
        longPlotSeconds,
        0,
    )
    signalData = signals.buildSigs(
        bufferData,
        sampleRate,
        longPlotSeconds,
        tempWindowSeconds,
        fftConfig,
    )
    return signalData


# Orchestrates CSV input through buffer, signals, ML, then charting.
def runCSV(csvPath, modelPath):
    dataFrame = pd.read_csv(csvPath)
    startRows = csvPlotStartRows(
        dataFrame.shape[0],
        sampleRate,
        bufferSeconds,
        periodSeconds,
        csvPlotPeriodCount,
    )
    csvChartDirs(outputDirPath)
    signalDataList = []
    for periodNumber, startRow in enumerate(startRows, start=1):
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
        signalData["mlData"] = mlData(signalData, modelPath)
        signalDataList.append(signalData)
        if periodNumber == 1:
            gui.printCsvSum(csvPath, sensorColumns, signalData)

        plotPaths = csvPlotPaths(outputDirPath, periodNumber)
        gui.plotSensor(signalData, 1, plotPaths["sensor1"], periodSeconds)
        print("plotSensor1Path:", plotPaths["sensor1"])
        gui.plotSensor(signalData, 2, plotPaths["sensor2"], periodSeconds)
        print("plotSensor2Path:", plotPaths["sensor2"])
        gui.plotAxisFftGrid(signalData, plotPaths["fftGrid"])
        print("plotAxisFftGridPath:", plotPaths["fftGrid"])

    longData = longSignalData(
        dataFrame,
        sampleRate,
        longPlotSeconds,
        tempWindowSeconds,
    )
    longPlotPaths = csvLongPlotPaths(outputDirPath)
    gui.plotSensor(
        longData,
        1,
        longPlotPaths["sensor1"],
        longPlotSeconds,
    )
    print("longSensor1Path:", longPlotPaths["sensor1"])
    gui.plotSensor(
        longData,
        2,
        longPlotPaths["sensor2"],
        longPlotSeconds,
    )
    print("longSensor2Path:", longPlotPaths["sensor2"])
    gui.plotAxisFftGrid(longData, longPlotPaths["fftGrid"])
    print("longAxisFftGridPath:", longPlotPaths["fftGrid"])

    return signalDataList


# Orchestrates live input through buffer, signals, and ML.
def runLive(
    port,
    baudrate,
    delimiter,
    timeout,
    encoding,
    modelPath,
):
    for bufferData in buffer.liveRoll(
        port,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        liveStepSeconds,
        baudrate,
        delimiter,
        timeout,
        encoding,
    ):
        signalData = signals.buildSigs(
            bufferData,
            sampleRate,
            periodSeconds,
            tempWindowSeconds,
            fftConfig,
        )
        signalData["mlData"] = mlData(signalData, modelPath)
        gui.printLiveSum(sensorColumns, signalData)


# Orchestrates labelled CSV training, model fitting, and weight saving.
def runTraining(trainingDataDirPath, modelPath):
    labelledCsvPaths = labelCsvPaths(trainingDataDirPath)
    trainingSet = ml.trainingSet(
        labelledCsvPaths,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        periodSeconds,
        trainingStepSeconds,
        tempWindowSeconds,
        fftConfig,
        featureNames,
        failureIndexes,
    )
    featureTensor = ml.featureTensor(trainingSet["featureMatrix"])
    labelTensor = ml.labelTensor(trainingSet["labelVector"])
    model = clfModel()
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
    probabilityDict = ml.probabilityDict(probabilityTensor, failureNames)
    predictedLabel = ml.predictedLabel(probabilityTensor, failureNames)
    trainingData = {
        "trainingSet": trainingSet,
        "lossHistory": lossHistory,
        "modelPath": modelPath,
        "probabilityDict": probabilityDict,
        "predictedLabel": predictedLabel,
    }
    gui.printTrainSum(trainingData)
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
