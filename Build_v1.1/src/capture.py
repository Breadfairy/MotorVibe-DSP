################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import csv
import math
import struct
import sys
import time

import serial

################################################################################
# variables/constants                                                          #
################################################################################
captureSec = 10
fileName = "test1.csv"
outDir = "ML_data"
port = "COM5"
baudRate = 1000000
timeout = 1.0
sampleRate = 1000.0
buffSampCount = 32
recFmt = "<I12hf"
accelScale = 16384.0
gyroScale = 131.0
readyPrefix = "Sample struct size (bytes):"
startCommand = b"START\n"
recSize = struct.calcsize(recFmt)
bufferSize = recSize * buffSampCount
csvColumns = [
    "t_us",
    "t_s",
    "ax1",
    "ay1",
    "az1",
    "gx1",
    "gy1",
    "gz1",
    "ax2",
    "ay2",
    "az2",
    "gx2",
    "gy2",
    "gz2",
    "tempC",
]

################################################################################
# helpers                                                                      #
################################################################################


def waitForReady(link):
    # Waits for the firmware text handshake before binary streaming starts.
    print("waiting for device ready...")
    while True:
        deviceLine = link.readline().decode("ascii", errors="ignore").strip()
        if len(deviceLine) == 0:
            continue
        print("device:", deviceLine)
        if deviceLine.startswith(readyPrefix):
            break
    link.write(startCommand)
    link.flush()


################################################################################
# main functions                                                               #
################################################################################


if len(sys.argv) > 1:
    fileName = sys.argv[1]
    if not fileName.endswith(".csv"):
        fileName += ".csv"

Path(outDir).mkdir(parents=True, exist_ok=True)

savePath = str(Path(outDir) / fileName)
rowCount = 0
leftOverBytes = b""
startTime = 0.0
firstTUsRaw = None

print(f"starting capture of {fileName} for {captureSec} seconds")
print(f"port: {port}")
print(f"baudRate: {baudRate}")
print(f"sampleRate: {sampleRate}")

with open(savePath, "w", newline="") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(csvColumns)

    with serial.Serial(
        port=port,
        baudrate=baudRate,
        timeout=timeout,
    ) as link:
        waitForReady(link)
        startTime = time.perf_counter()

        while (time.perf_counter() - startTime) < captureSec:
            newBytes = link.read(bufferSize)
            if len(newBytes) == 0:
                continue

            leftOverBytes += newBytes

            while len(leftOverBytes) >= recSize:
                packetBytes = leftOverBytes[:recSize]
                leftOverBytes = leftOverBytes[recSize:]
                packetValues = struct.unpack(recFmt, packetBytes)
                tUsRaw = packetValues[0]
                if firstTUsRaw is None:
                    firstTUsRaw = tUsRaw
                tUs = (tUsRaw - firstTUsRaw) & 0xFFFFFFFF
                tempC = packetValues[13]
                if not math.isfinite(tempC):
                    tempC = 0.0
                rowValue = [
                    tUs,
                    tUs / 1e6,
                    packetValues[1] / accelScale,
                    packetValues[2] / accelScale,
                    packetValues[3] / accelScale,
                    packetValues[4] / gyroScale,
                    packetValues[5] / gyroScale,
                    packetValues[6] / gyroScale,
                    packetValues[7] / accelScale,
                    packetValues[8] / accelScale,
                    packetValues[9] / accelScale,
                    packetValues[10] / gyroScale,
                    packetValues[11] / gyroScale,
                    packetValues[12] / gyroScale,
                    tempC,
                ]
                writer.writerow(rowValue)
                rowCount += 1

runSecs = time.perf_counter() - startTime
avgRate = rowCount / runSecs

print("capture complete")
print("rows logged:", rowCount)
print("avg sample rate:", avgRate)
print("saved to:", savePath)
