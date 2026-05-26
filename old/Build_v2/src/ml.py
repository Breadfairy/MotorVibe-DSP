################################################################################
# Imports                                                                      #
################################################################################
import joblib
import numpy as np
import pandas as pd

import data
import signals

featureNamesPerGroup = [
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


# Builds the eight compact features for one sensor group.
def featuresFromGroup(groupName, timeSignals, frequencyFeatures):
    magnitude = timeSignals[f"{groupName}Mag"]
    return np.array(
        [
            float(np.sqrt(np.mean(magnitude * magnitude))),
            float(np.min(magnitude)),
            float(np.max(magnitude)),
            frequencyFeatures[f"{groupName}FundHz"],
            frequencyFeatures[f"{groupName}FundMag"],
            frequencyFeatures[f"{groupName}LowOrderRms"],
            frequencyFeatures[f"{groupName}BpfoRms"],
            frequencyFeatures[f"{groupName}BpfiRms"],
        ],
        dtype=np.float32,
    )


# Builds the compact 32-feature row from derived signal dictionaries.
def featuresFromSignals(timeSignals, frequencyFeatures):
    groupRows = []
    for groupName, _colX, _colY, _colZ in signals.axisGroups:
        groupRows.append(
            featuresFromGroup(groupName, timeSignals, frequencyFeatures)
        )
    return np.concatenate(groupRows).astype(np.float32)


# Lists feature names in model order.
def featureNames():
    names = []
    for groupName, _colX, _colY, _colZ in signals.axisGroups:
        for featureName in featureNamesPerGroup:
            names.append(f"{groupName}_{featureName}")
    return names


# Builds one 32-feature summary from one 1-second window.
def featuresFromWindow(oneSecondWindow, sampleRate):
    timeSignals = signals.mlTimeSignals(oneSecondWindow, sampleRate)
    frequencyFeatures = signals.mlFreqSignals(
        timeSignals,
        sampleRate,
        signals.fftConfig,
    )
    return featuresFromSignals(timeSignals, frequencyFeatures)


# Gets the strongest label.
def stateFromProb(probabilities, labelNames):
    topIndex = int(np.argmax(probabilities))
    state = labelNames[topIndex]
    confidence = float(probabilities[topIndex])
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

    def featureMatrixFromCapture(self, capture, clean=True):
        if clean:
            capture = data.cleanCapture(capture)

        windows = data.windowsFromCapture(
            capture,
            self.sampleRate,
            self.winSecs,
            self.stepSecs,
        )

        allWindowFeatures = []
        for oneSecondWindow in windows:
            allWindowFeatures.append(
                featuresFromWindow(oneSecondWindow, self.sampleRate)
            )
        return np.vstack(allWindowFeatures).astype(np.float32)

    def featureMatrixFromCsv(self, csvPath):
        capture = data.readCsv(csvPath)
        return self.featureMatrixFromCapture(capture, clean=True)

    def featuresFromSamples(self, samples):
        capture = pd.DataFrame(samples, columns=data.signalCols)
        capture = data.cleanCapture(capture)
        return featuresFromWindow(capture, self.sampleRate)

    def predictSamples(self, samples):
        windowFeatures = self.featuresFromSamples(samples)
        return self.model.predict_proba([windowFeatures])[0]

    def predictCsv(self, csvPath):
        featureMatrix = self.featureMatrixFromCsv(csvPath)
        probabilitiesByWindow = self.model.predict_proba(featureMatrix)
        meanProbabilities = probabilitiesByWindow.mean(axis=0)
        state, confidence = stateFromProb(meanProbabilities, self.labelNames)
        return state, confidence, meanProbabilities

    # Backwards-compatible method names used by older scripts in this build.
    featureMatrixFromFrame = featureMatrixFromCapture
    windowFeaturesFromSamples = featuresFromSamples
    featureRowFromRows = featuresFromSamples
    predictRows = predictSamples


# Backwards-compatible function names used by older scripts in this build.
groupWindowFeatures = featuresFromGroup
groupFeatureRow = featuresFromGroup
windowFeaturesFromSignals = featuresFromSignals
featureRowFromSignals = featuresFromSignals
windowFeaturesFromWindow = featuresFromWindow
featureRowFromWindow = featuresFromWindow
