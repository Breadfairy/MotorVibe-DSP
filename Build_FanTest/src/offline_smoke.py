################################################################################
# Imports                                                                      #
################################################################################
import math
import struct

import pandas as pd

import ml_core
import signals_core

################################################################################
# variables/constants                                                          #
################################################################################
packetCount = 1000
sampleRate = 1000.0
bufferSampleCount = 32
recordFormat = "<I6h"
accelScale = 16384.0
gyroScale = 131.0
recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount
csvColumns = [
    "t_us",
    "t_s",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]

################################################################################
# helpers                                                                      #
################################################################################


# Converts one unpacked serial packet into the shared CSV row format.
def packetRow(packetValues):
    rowValue = [
        packetValues[0],
        packetValues[0] / 1e6,
        packetValues[1] / accelScale,
        packetValues[2] / accelScale,
        packetValues[3] / accelScale,
        packetValues[4] / gyroScale,
        packetValues[5] / gyroScale,
        packetValues[6] / gyroScale,
    ]
    return rowValue


################################################################################
# main functions                                                               #
################################################################################

print("recordFormat:", recordFormat)
print("recordSize:", recordSize)
print("bufferSize:", bufferSize)

assert recordSize == 16
assert bufferSize == 512

payload = bytearray()
for packetIndex in range(packetCount):
    tUs = packetIndex * 1000
    ax = int(1000 * math.sin(packetIndex * 0.02))
    ay = int(900 * math.sin(packetIndex * 0.03))
    az = int(1200 * math.sin(packetIndex * 0.04))
    gx = int(500 * math.sin(packetIndex * 0.01))
    gy = int(450 * math.sin(packetIndex * 0.05))
    gz = int(350 * math.sin(packetIndex * 0.06))
    payload.extend(
        struct.pack(
            recordFormat,
            tUs,
            ax,
            ay,
            az,
            gx,
            gy,
            gz,
        )
    )

chunkSizes = [7, 13, 64, 101, 257, 509]
leftOver = b""
rows = []
byteIndex = 0
chunkIndex = 0
while byteIndex < len(payload):
    chunkSize = chunkSizes[chunkIndex % len(chunkSizes)]
    chunkIndex += 1
    nextByteIndex = min(byteIndex + chunkSize, len(payload))
    leftOver += payload[byteIndex:nextByteIndex]
    byteIndex = nextByteIndex
    while len(leftOver) >= recordSize:
        packetBytes = leftOver[:recordSize]
        leftOver = leftOver[recordSize:]
        packetValues = struct.unpack(recordFormat, packetBytes)
        rows.append(packetRow(packetValues))

assert len(rows) == packetCount
assert len(leftOver) == 0

dataFrame = pd.DataFrame(rows, columns=csvColumns)
rawSignals = signals_core.rawArrays(dataFrame)
timeSignals = signals_core.timeData(rawSignals, sampleRate)
freqSignals = signals_core.freqData(
    rawSignals,
    sampleRate,
    signals_core.fftConfig,
)
featureVector = ml_core.featureVector(timeSignals, freqSignals)

assert featureVector.shape == (12,)

print("packetRows:", len(rows))
print("featureShape:", featureVector.shape)
print("featureHead:", featureVector[:6])
print("offlineSmoke: PASS")
