################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import sys
import time

import pandas as pd

import buffer
import ml
import rules
import signals

################################################################################
# variables/constants                                                          #
################################################################################
rootDir = Path(__file__).resolve().parents[1]
mode = "sixaxis_1k"
port = None
alertProb = 0.60

################################################################################
# helpers                                                                      #
################################################################################


# Renders one simple terminal block in place.
def renderBlock(lines):
    text = "\n".join(lines)
    print(f"\x1b[2J\x1b[H{text}", end="", flush=True)

################################################################################
# main functions                                                               #
################################################################################


if len(sys.argv) > 1:
    mode = sys.argv[1]
if len(sys.argv) > 2:
    port = sys.argv[2]

modelPath = rootDir / "outputs" / "models" / f"{mode}.pth"
loadMode, net = ml.loadModel(modelPath)
if loadMode != mode:
    raise ValueError("model mode mismatch")

rows = []
rowCount = 0
newRows = 0
firstFrame = True
t0 = 0.0
lastSample = 0
winRows = ml.windowRows(mode)
stepRows = ml.stepRows(mode)

print(f"mode: {mode}")
print(f"port: {port or buffer.detectPort()}")
print(f"baudRate: {buffer.streamBaud}")
print(f"sampleRate: {buffer.sampleRate(mode)}")
print(f"modelPath: {modelPath}")

with buffer.openLink(port) as link:
    buffer.waitForReady(link)
    buffer.stopStream(link)
    buffer.sendMode(link, mode)
    buffer.startStream(link)
    t0 = time.perf_counter()
    while True:
        head, payload = buffer.readFrame(link, mode, firstFrame)
        firstFrame = False
        batchRows = buffer.decodeRows(head, payload)
        rows += batchRows
        rowCount += len(batchRows)
        newRows += len(batchRows)
        lastSample = batchRows[-1][0]
        if len(rows) > winRows:
            rows = rows[-winRows:]
        if len(rows) < winRows:
            continue
        if newRows < stepRows:
            continue
        newRows = 0
        dataFrame = pd.DataFrame(rows, columns=buffer.csvCols(mode))
        sig = signals.buildSignals(dataFrame, mode)
        vec = ml.modelVector(sig)
        x = ml.featureTensor(vec[None, :])
        prob = ml.runModel(net, x)
        probRows = ml.probDict(prob)
        top = ml.topLabel(prob)
        topProb = probRows[top]
        ruleRows = rules.runRules(sig)
        state = "watch"
        text = ruleRows["text"]
        if top == "healthy" and topProb >= alertProb:
            state = "healthy"
            text = "healthy signature dominant"
        if top != "healthy" and topProb >= alertProb:
            state = "warning"
            text = f"{ml.displayLabel(top)} signature rising"
        runSecs = time.perf_counter() - t0
        avgRate = rowCount / runSecs
        lines = [
            "Live Monitor",
            "",
            f"mode: {mode}",
            f"elapsedSeconds: {runSecs:.2f}",
            f"windowRows: {winRows}",
            f"avgSampleRate: {avgRate:.1f}",
            f"lastSample: {lastSample}",
            f"sequence: {head['sequence']}",
            "",
            f"status: {state}",
            f"rules: {ruleRows['state']}",
            f"message: {text}",
            f"topClass: {ml.displayLabel(top)}",
            f"topProb: {topProb * 100.0:.1f}%",
            "",
            f"fundHz: {sig['fundHz']:.3f}",
            f"fundMag: {sig['fundMag']:.3f}",
            f"bpfoRms: {sig['bpfoRms']:.3f}",
            f"bpfiRms: {sig['bpfiRms']:.3f}",
            "",
            "Class Probabilities",
        ]
        for i in ml.labelNames:
            lines.append(
                f"{ml.displayLabel(i)}: {probRows[i] * 100.0:.1f}%"
            )
        renderBlock(lines)
