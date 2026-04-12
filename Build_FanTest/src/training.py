################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path

import numpy as np
import pandas as pd

import ml_core
import signals_core

################################################################################
# variables/constants                                                          #
################################################################################
trainDir = "data/train"
modelPath = "outputs/models/simpleMotorClassifier.pth"
sampleRate = 1000.0
winSecs = 1.0
stepSecs = 1.0
epochs = 200
learnRate = 0.001

################################################################################
# helpers                                                                      #
################################################################################


# Collects labelled CSV paths from the training directory layout.
def labelCsvs(trainDir):
    csvData = {}
    for labelName in ml_core.labelNames:
        labelDir = Path(trainDir) / labelName
        csvPaths = [str(csvPath) for csvPath in sorted(labelDir.glob("*.csv"))]
        if len(csvPaths) > 0:
            csvData[labelName] = csvPaths
    return csvData


# Builds the valid window start rows for one dataframe.
def startRows(rowCount, sampleRate, winSecs, stepSecs):
    winRows = int(sampleRate * winSecs)
    stepSize = int(sampleRate * stepSecs)
    lastRow = rowCount - winRows
    rowStarts = []
    for startRow in range(0, lastRow + 1, stepSize):
        rowStarts.append(startRow)
    return rowStarts


################################################################################
# main functions                                                               #
################################################################################


# Runs the shared one-second feature extraction and model training flow.
def main():
    labelledCsvs = labelCsvs(trainDir)
    featureRows = []
    labelRows = []
    datasetMeta = []
    winRows = int(sampleRate * winSecs)

    for labelName, csvPaths in labelledCsvs.items():
        for csvPath in csvPaths:
            dataFrame = pd.read_csv(csvPath)
            csvStarts = startRows(
                dataFrame.shape[0],
                sampleRate,
                winSecs,
                stepSecs,
            )
            for startRow in csvStarts:
                windowFrame = dataFrame.iloc[
                    startRow : startRow + winRows
                ].reset_index(drop=True)
                rawSignals = signals_core.rawArrays(windowFrame)
                timeSignals = signals_core.timeData(rawSignals, sampleRate)
                freqSignals = signals_core.freqData(
                    rawSignals,
                    sampleRate,
                    signals_core.fftConfig,
                )
                featureRows.append(
                    ml_core.featureVector(timeSignals, freqSignals)
                )
                labelRows.append(labelName)
            datasetMeta.append(
                {
                    "labelName": labelName,
                    "csvPath": csvPath,
                    "windowCount": len(csvStarts),
                }
            )

    featureMatrix = np.vstack(featureRows)
    labelVector = ml_core.encodeLabels(labelRows)
    featureData = ml_core.featureTensor(featureMatrix)
    labelData = ml_core.labelTensor(labelVector)
    modelValue = ml_core.model(ml_core.hiddenSize)
    lossHistory = ml_core.trainModel(
        modelValue,
        featureData,
        labelData,
        epochs,
        learnRate,
    )

    Path(modelPath).parent.mkdir(parents=True, exist_ok=True)
    ml_core.saveModel(modelValue, modelPath)

    probTensor = ml_core.runModel(modelValue, featureData[:1])
    print("featureShape:", featureMatrix.shape)
    print("labelShape:", labelVector.shape)
    print("datasetMeta:", datasetMeta)
    print("modelPath:", modelPath)
    print("lossStart:", lossHistory[0])
    print("lossEnd:", lossHistory[-1])
    print("topLabel:", ml_core.topLabel(probTensor))
    print("probDict:", ml_core.probDict(probTensor))


if __name__ == "__main__":
    main()
