################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import csv
import sys
import time

import buffer

################################################################################
# variables/constants                                                          #
################################################################################
rootDir = Path(__file__).resolve().parents[1]
captureSecs = 10
fileName = "test1.csv"
mode = "sixaxis_1k"
port = None
outDir = rootDir / "data" / "capture"

################################################################################
# main functions                                                               #
################################################################################


if len(sys.argv) > 1:
    fileName = sys.argv[1]
if len(sys.argv) > 2:
    mode = sys.argv[2]
if len(sys.argv) > 3:
    port = sys.argv[3]
if not fileName.endswith(".csv"):
    fileName += ".csv"

Path(outDir).mkdir(parents=True, exist_ok=True)
savePath = outDir / fileName
rowCount = 0
firstFrame = True
t0 = 0.0

print(f"fileName: {fileName}")
print(f"mode: {mode}")
print(f"port: {port or buffer.detectPort()}")
print(f"baudRate: {buffer.streamBaud}")
print(f"sampleRate: {buffer.sampleRate(mode)}")

with open(savePath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(buffer.csvCols(mode))
    with buffer.openLink(port) as link:
        buffer.waitForReady(link)
        buffer.stopStream(link)
        buffer.sendMode(link, mode)
        buffer.startStream(link)
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < captureSecs:
            head, payload = buffer.readFrame(link, mode, firstFrame)
            firstFrame = False
            rows = buffer.decodeRows(head, payload)
            for i in rows:
                writer.writerow(i)
            rowCount += len(rows)

runSecs = time.perf_counter() - t0
avgRate = rowCount / runSecs

print("capture complete")
print("rows:", rowCount)
print("avgRate:", avgRate)
print("savePath:", savePath)
