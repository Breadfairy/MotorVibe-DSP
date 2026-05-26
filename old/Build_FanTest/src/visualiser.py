################################################################################
# Imports                                                                      #
################################################################################
import os
import sys

import pandas as pd

import buffer

if "BUILD_FANTEST_MPL_BACKEND" not in os.environ:
    if sys.platform == "darwin":
        os.environ["BUILD_FANTEST_MPL_BACKEND"] = "MacOSX"
    elif os.name == "nt":
        os.environ["BUILD_FANTEST_MPL_BACKEND"] = "TkAgg"
    else:
        os.environ["BUILD_FANTEST_MPL_BACKEND"] = "TkAgg"

import charting
import signals

################################################################################
# variables/constants                                                          #
################################################################################
mode = "all_1k"
port = None
plotSecs = {
    "all_1k": 2.0,
    "gyro_8k": 1.0,
}
visSecs = 0.1

################################################################################
# main functions                                                               #
################################################################################


if len(sys.argv) > 1:
    mode = sys.argv[1]
if len(sys.argv) > 2:
    port = sys.argv[2]

plotRows = int(buffer.sampleRate(mode) * plotSecs[mode])
visRows = max(1, int(buffer.sampleRate(mode) * visSecs))
rows = []
newRows = 0
firstFrame = True

print(f"mode: {mode}")
print(f"port: {port or buffer.detectPort()}")
print(f"baudRate: {buffer.streamBaud}")
print(f"sampleRate: {buffer.sampleRate(mode)}")
print(f"refreshRows: {visRows}")

liveFig = charting.buildLiveFig(mode, plotRows)

with buffer.openLink(port) as link:
    buffer.waitForReady(link)
    buffer.stopStream(link)
    buffer.sendMode(link, mode)
    buffer.startStream(link)
    while True:
        head, payload = buffer.readFrame(link, mode, firstFrame)
        firstFrame = False
        batchRows = buffer.decodeRows(head, payload)
        rows += batchRows
        newRows += len(batchRows)
        if len(rows) > plotRows:
            rows = rows[-plotRows:]
        if len(rows) < 64:
            continue
        if newRows < visRows:
            continue
        newRows = 0
        dataFrame = pd.DataFrame(rows, columns=buffer.csvCols(mode))
        sig = signals.buildSignals(dataFrame, mode)
        charting.updateLiveFig(liveFig, sig)
