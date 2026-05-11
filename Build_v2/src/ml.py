################################################################################
# Imports                                                                      #
################################################################################
import joblib
import numpy as np
import pandas as pd

import data
import signals

groupFeatureNames = [
    "rms",
    "min",
    "max",
    "fundHz",
    "fundMag",
    "lowOrderRms",
    "bpfoRms",
    "bpfiRms",
]

################################################################################
# helpers                                                                      #
################################################################################


# Builds the compact model feature row.
def modelInput(timeData, freqData):
    parts = []
    for groupName, colX, colY, colZ in signals.axisGroups:
        mag = timeData[f"{groupName}Mag"]
        stats = [
            float(np.sqrt(np.mean(mag * mag))),
            float(np.min(mag)),
            float(np.max(mag)),
            freqData[f"{groupName}FundHz"],
            freqData[f"{groupName}FundMag"],
            freqData[f"{groupName}LowOrderRms"],
            freqData[f"{groupName}BpfoRms"],
            freqData[f"{groupName}BpfiRms"],
        ]
        parts.append(np.array(stats, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


# Lists feature names in model order.
def featureNames():
    names = []
    for groupName, colX, colY, colZ in signals.axisGroups:
        for featureName in groupFeatureNames:
            names.append(f"{groupName}_{featureName}")
    return names


# Builds one feature row from one window.
def featureRowFromWindow(windowFrame, sampleRate):
    timeData = signals.mlTimeSignals(windowFrame, sampleRate)
    freqData = signals.mlFreqSignals(timeData, sampleRate, signals.fftConfig)
    return modelInput(timeData, freqData)


# Gets the strongest label.
def stateFromProb(probVector, labelNames):
    topIndex = int(np.argmax(probVector))
    state = labelNames[topIndex]
    confidence = float(probVector[topIndex])
    return state, confidence

################################################################################
# main classes                                                                 #
################################################################################


class MotorPumpClassifier:
    def __init__(self, bundle):
        self.bundle = bundle
        self.model = bundle["model"]
        self.labelNames = bundle["labelNames"]
        self.sampleRate = bundle["sampleRate"]
        self.winSecs = bundle["winSecs"]
        self.stepSecs = bundle["stepSecs"]

    @classmethod
    def load(cls, modelPath):
        bundle = joblib.load(modelPath)
        return cls(bundle)

    def featureMatrixFromFrame(self, dataFrame, clean=True):
        df = dataFrame
        if clean:
            df = data.cleanFrame(df)
        windows = data.windowFrames(
            df,
            self.sampleRate,
            self.winSecs,
            self.stepSecs,
        )

        featureRows = []
        for windowFrame in windows:
            featureRows.append(featureRowFromWindow(windowFrame, self.sampleRate))
        return np.vstack(featureRows).astype(np.float32)

    def featureMatrixFromCsv(self, csvPath):
        df = data.readCsv(csvPath)
        return self.featureMatrixFromFrame(df, clean=True)

    def featureRowFromRows(self, rows):
        df = pd.DataFrame(rows, columns=data.signalCols)
        df = data.cleanFrame(df)
        return featureRowFromWindow(df, self.sampleRate)

    def predictRows(self, rows):
        featureRow = self.featureRowFromRows(rows)
        return self.model.predict_proba([featureRow])[0]

    def predictCsv(self, csvPath):
        featureMatrix = self.featureMatrixFromCsv(csvPath)
        probMatrix = self.model.predict_proba(featureMatrix)
        meanProb = probMatrix.mean(axis=0)
        state, confidence = stateFromProb(meanProb, self.labelNames)
        return state, confidence, meanProb
