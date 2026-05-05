################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
buildDir = Path(__file__).resolve().parents[1]
modelPath = buildDir / "outputs" / "models" / "rawAxisClassifier.pth"
inputPath = buildDir / "data" / "training" / "main"

################################################################################
# main functions                                                               #
################################################################################


def loadBundle(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    sourcePath = inputPath
    if len(sys.argv) > 1:
        sourcePath = Path(sys.argv[1])

    if sourcePath.is_file():
        csvList = [sourcePath]
    else:
        csvList = sourcePath.glob("*.csv")

    bundle = loadBundle(modelPath)
    modelValue = nn.Linear(
        bundle["inputSize"],
        len(bundle["labelNames"]),
    )
    modelValue.load_state_dict(bundle["stateDict"])
    modelValue.eval()

    featureMean = bundle["featureMean"]
    featureStd = bundle["featureStd"]
    sampleRate = bundle["sampleRate"]
    winSecs = bundle["winSecs"]
    stepSecs = bundle["stepSecs"]
    labelNames = bundle["labelNames"]

    for csvPath in csvList:
        df = data.readCsv(csvPath)
        df = data.cleanFrame(df)
        windows = data.windowFrames(
            df,
            sampleRate,
            winSecs,
            stepSecs,
        )

        featureRows = []
        for windowFrame in windows:
            x = signals.timeSignals(windowFrame, sampleRate)
            y = signals.fftSignals(x, sampleRate, signals.fftConfig)
            featureRows.append(signals.modelInput(x, y))

        featureMatrix = np.vstack(featureRows).astype(np.float32)
        scaledMatrix = (featureMatrix - featureMean) / featureStd
        featureTensor = torch.tensor(scaledMatrix, dtype=torch.float32)

        with torch.no_grad():
            outputTensor = modelValue(featureTensor)
            probTensor = torch.softmax(outputTensor, dim=1)
            probMatrix = probTensor.detach().cpu().numpy()

        meanProb = probMatrix.mean(axis=0)
        topIndex = int(np.argmax(meanProb))

        print("csvPath:", csvPath)
        print("topLabel:", labelNames[topIndex])
        print("topProb:", float(meanProb[topIndex]))
        print("")


if __name__ == "__main__":
    main()
