################################################################################
# Imports                                                                      #
################################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import buffer

################################################################################
# variables/constants                                                          #
################################################################################
labelNames = [
    "healthy",
    "tiltX",
    "tiltY",
    "looseScrew1",
    "looseScrew2",
    "looseScrew3",
    "looseScrew4",
    "imbalanceAdded",
    "rubbingContact",
]
winSecs = {
    "sixaxis_1k": 1.0,
    "gyro_8k": 1.0,
}
stepSecs = {
    "sixaxis_1k": 0.25,
    "gyro_8k": 0.25,
}
hiddenSize = 64

################################################################################
# helpers                                                                      #
################################################################################


# Converts one compact label name into a readable label.
def displayLabel(name):
    text = []
    prev = False
    for i in name:
        if i in ["_", "-"]:
            text.append(" ")
            prev = False
            continue
        if i.isupper() and prev:
            text.append(" ")
        text.append(i)
        prev = i.islower() or i.isdigit()
    return "".join(text).strip().title()


# Builds one stable row count from the current mode and window seconds.
def windowRows(mode):
    return int(buffer.sampleRate(mode) * winSecs[mode])


# Builds one stable step row count from the current mode and step seconds.
def stepRows(mode):
    return int(buffer.sampleRate(mode) * stepSecs[mode])


# Standardises one vector for direct model input.
def norm(x):
    y = np.asarray(x, dtype=np.float32)
    y = y - np.mean(y)
    s = float(np.std(y))
    if s > 0.0:
        y = y / s
    return y


# Encodes label names into integer indexes.
def encodeLabels(names):
    rows = []
    for i in names:
        rows.append(labelNames.index(i))
    return np.array(rows, dtype=np.int64)


# Converts one feature matrix into a float tensor.
def featureTensor(x):
    return torch.tensor(x, dtype=torch.float32)


# Converts one label vector into a long tensor.
def labelTensor(x):
    return torch.tensor(x, dtype=torch.long)

################################################################################
# classes or main functions                                                    #
################################################################################


# Builds one direct dense classifier for the selected stream mode.
class MotorModel(nn.Module):
    # Builds the current dense stack for one direct stream vector.
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, outputSize),
        )

    # Runs one forward pass through the classifier.
    def forward(self, x):
        return self.net(x)


# Builds one direct stream vector from raw time and raw FFT axes.
def modelVector(sig):
    rows = []
    for i in sig["timeCols"]:
        rows.append(norm(sig[i]))
    for i in sig["fftCols"]:
        rows.append(norm(sig[f"{i}Fft"]))
    return np.concatenate(rows).astype(np.float32)


# Builds one model instance for the selected mode.
def model(mode):
    n = windowRows(mode)
    m = (n // 2) + 1
    k = len(buffer.timeCols(mode))
    inputSize = (n * k) + (m * k)
    return MotorModel(
        inputSize,
        hiddenSize,
        len(labelNames),
    )


# Trains one model from the current feature and label tensors.
def trainModel(net, x, y, epochs, learnRate):
    lossFn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=learnRate)
    lossRows = []
    for i in range(epochs):
        opt.zero_grad()
        out = net(x)
        loss = lossFn(out, y)
        loss.backward()
        opt.step()
        lossRows.append(float(loss.item()))
    return lossRows


# Runs one inference pass and returns probabilities.
def runModel(net, x):
    out = net(x)
    return torch.softmax(out, dim=1)


# Builds one named probability dictionary from one inference row.
def probDict(prob):
    row = prob.detach().cpu().numpy()[0]
    out = {}
    for i, j in enumerate(labelNames):
        out[j] = float(row[i])
    return out


# Returns the highest-probability label name.
def topLabel(prob):
    row = prob.detach().cpu().numpy()[0]
    i = int(np.argmax(row))
    return labelNames[i]


# Saves one mode-specific model package to disk.
def saveModel(net, mode, modelPath):
    pack = {
        "mode": mode,
        "state": net.state_dict(),
    }
    torch.save(pack, modelPath)


# Loads one mode-specific model package from disk.
def loadModel(modelPath):
    pack = torch.load(modelPath, map_location="cpu")
    mode = pack["mode"]
    net = model(mode)
    net.load_state_dict(pack["state"])
    return mode, net
