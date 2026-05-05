################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
buildDir = Path(__file__).resolve().parents[1]
trainDir = buildDir / "data" / "training" / "main"
modelPath = buildDir / "outputs" / "models" / "rawAxisClassifier.pth"
classNames = ["good", "leak", "inflow", "outflow", "temp", "tilt"]
sampleRate = 1000.0
winSecs = 1.0
stepSecs = 1.0
trainFrac = 0.25
valFrac = 0.25
epochs = 200
learnRate = 0.001

################################################################################
# main functions                                                               #
################################################################################


def main():
    featureRows = []
    classRows = []
    valFeatureRows = []
    valClassRows = []

    for className in classNames:
        if className == "good": classIndex = 0
        elif className == "leak": classIndex = 1
        elif className == "inflow": classIndex = 2
        elif className == "outflow": classIndex = 3
        elif className == "temp": classIndex = 4
        elif className == "tilt": classIndex = 5
        for csvPath in trainDir.glob(f"{className}*.csv"):
            df = data.readCsv(csvPath)
            df = data.cleanFrame(df)
            windows = data.windowFrames(
                df,
                sampleRate,
                winSecs,
                stepSecs,
            )
            trainEnd = int(len(windows) * trainFrac)
            valEnd = int(len(windows) * (trainFrac + valFrac))

            for windowFrame in windows[:trainEnd]:
                x = signals.timeSignals(windowFrame, sampleRate)
                y = signals.fftSignals(x, sampleRate, signals.fftConfig)
                featureRows.append(signals.modelInput(x, y))
                classRows.append(classIndex)

            for windowFrame in windows[trainEnd:valEnd]:
                x = signals.timeSignals(windowFrame, sampleRate)
                y = signals.fftSignals(x, sampleRate, signals.fftConfig)
                valFeatureRows.append(signals.modelInput(x, y))
                valClassRows.append(classIndex)

    featureMatrix = np.vstack(featureRows).astype(np.float32)
    classVector = np.array(classRows, dtype=np.int64)
    valFeatureMatrix = np.vstack(valFeatureRows).astype(np.float32)
    valClassVector = np.array(valClassRows, dtype=np.int64)

    featureMean = featureMatrix.mean(axis=0).astype(np.float32)
    featureStd = featureMatrix.std(axis=0).astype(np.float32)
    featureStd[featureStd == 0.0] = 1.0
    scaledMatrix = (featureMatrix - featureMean) / featureStd
    valScaledMatrix = (valFeatureMatrix - featureMean) / featureStd

    featureTensor = torch.tensor(scaledMatrix, dtype=torch.float32)
    classTensor = torch.tensor(classVector, dtype=torch.long)
    valFeatureTensor = torch.tensor(valScaledMatrix, dtype=torch.float32)
    valClassTensor = torch.tensor(valClassVector, dtype=torch.long)

    modelValue = nn.Linear(
        featureTensor.shape[1],
        len(classNames),
    )

    lossFn = nn.CrossEntropyLoss()
    optimiser = optim.Adam(modelValue.parameters(), lr=learnRate)
    lossHistory = []

    for i in range(epochs):
        optimiser.zero_grad()
        outputTensor = modelValue(featureTensor)
        lossValue = lossFn(outputTensor, classTensor)
        lossValue.backward()
        optimiser.step()
        lossHistory.append(float(lossValue.item()))

    with torch.no_grad():
        outputTensor = modelValue(featureTensor)
        predVector = torch.argmax(outputTensor, dim=1)
        trainAcc = float((predVector == classTensor).float().mean().item())
        valOutputTensor = modelValue(valFeatureTensor)
        valPredVector = torch.argmax(valOutputTensor, dim=1)
        valAcc = float((valPredVector == valClassTensor).float().mean().item())

    modelPath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stateDict": modelValue.state_dict(),
            "featureMean": featureMean,
            "featureStd": featureStd,
            "labelNames": classNames,
            "sampleRate": sampleRate,
            "winSecs": winSecs,
            "stepSecs": stepSecs,
            "trainFrac": trainFrac,
            "valFrac": valFrac,
            "inputSize": int(featureTensor.shape[1]),
        },
        modelPath,)

    print("lossStart:", f"{lossHistory[0]:.2f}")
    print("lossEnd:", f"{lossHistory[-1]:.2f}")
    print("trainAcc:", f"{trainAcc * 100.0:.2f}%")
    print("valAcc:", f"{valAcc * 100.0:.2f}%")


if __name__ == "__main__":
    main()
