################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import sys

import numpy as np

import buffer
import ml
import signals

################################################################################
# variables/constants                                                          #
################################################################################
rootDir = Path(__file__).resolve().parents[1]
mode = "sixaxis_1k"
epochs = 200
learnRate = 0.001

################################################################################
# helpers                                                                      #
################################################################################


# Collects labelled CSV paths from the current mode training directory.
def labelCsvs(trainDir):
    rows = {}
    for i in ml.labelNames:
        path = trainDir / i
        files = [j for j in sorted(path.glob("*.csv"))]
        if len(files) > 0:
            rows[i] = files
    return rows


# Builds the current window start rows for one dataframe.
def startRows(rowCount, winRows, stepRows):
    out = []
    last = rowCount - winRows
    for i in range(0, last + 1, stepRows):
        out.append(i)
    return out

################################################################################
# main functions                                                               #
################################################################################


if len(sys.argv) > 1:
    mode = sys.argv[1]

trainDir = rootDir / "data" / "train" / mode
modelPath = rootDir / "outputs" / "models" / f"{mode}.pth"
winRows = ml.windowRows(mode)
stepRows = ml.stepRows(mode)
fileRows = labelCsvs(trainDir)
xRows = []
yRows = []
metaRows = []

for i in fileRows:
    for j in fileRows[i]:
        dataFrame = signals.readCSV(j)
        starts = startRows(dataFrame.shape[0], winRows, stepRows)
        for k in starts:
            winFrame = dataFrame.iloc[k : k + winRows].reset_index(drop=True)
            sig = signals.buildSignals(winFrame, mode)
            xRows.append(ml.modelVector(sig))
            yRows.append(i)
        metaRows.append(
            {
                "label": i,
                "csv": str(j),
                "windows": len(starts),
            }
        )

x = np.vstack(xRows)
y = ml.encodeLabels(yRows)
xData = ml.featureTensor(x)
yData = ml.labelTensor(y)
net = ml.model(mode)
lossRows = ml.trainModel(net, xData, yData, epochs, learnRate)
Path(modelPath).parent.mkdir(parents=True, exist_ok=True)
ml.saveModel(net, mode, modelPath)
prob = ml.runModel(net, xData[:1])

print("mode:", mode)
print("trainDir:", trainDir)
print("featureShape:", x.shape)
print("labelShape:", y.shape)
print("metaRows:", metaRows)
print("modelPath:", modelPath)
print("lossStart:", lossRows[0])
print("lossEnd:", lossRows[-1])
print("topLabel:", ml.topLabel(prob))
print("probDict:", ml.probDict(prob))
