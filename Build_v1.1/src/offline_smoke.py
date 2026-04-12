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
recordFormat = "<I12hf"
accelScale = 16384.0
gyroScale = 131.0
recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount
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
tempWindowSeconds = 0.01

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
        packetValues[7] / accelScale,
        packetValues[8] / accelScale,
        packetValues[9] / accelScale,
        packetValues[10] / gyroScale,
        packetValues[11] / gyroScale,
        packetValues[12] / gyroScale,
        packetValues[13],
    ]
    return rowValue


################################################################################
# main functions                                                               #
################################################################################

print("recordFormat:", recordFormat)
print("recordSize:", recordSize)
print("bufferSize:", bufferSize)

assert recordSize == 32
assert bufferSize == 1024

payload = bytearray()
for packetIndex in range(packetCount):
    tUs = packetIndex * 1000
    ax1 = int(1000 * math.sin(packetIndex * 0.02))
    ay1 = int(900 * math.sin(packetIndex * 0.03))
    az1 = int(1200 * math.sin(packetIndex * 0.04))
    gx1 = int(500 * math.sin(packetIndex * 0.01))
    gy1 = int(450 * math.sin(packetIndex * 0.05))
    gz1 = int(350 * math.sin(packetIndex * 0.06))
    ax2 = int(1100 * math.sin(packetIndex * 0.02 + 0.2))
    ay2 = int(950 * math.sin(packetIndex * 0.03 + 0.2))
    az2 = int(1250 * math.sin(packetIndex * 0.04 + 0.2))
    gx2 = int(520 * math.sin(packetIndex * 0.01 + 0.1))
    gy2 = int(470 * math.sin(packetIndex * 0.05 + 0.1))
    gz2 = int(360 * math.sin(packetIndex * 0.06 + 0.1))
    tempC = 36.5 + 0.3 * math.sin(packetIndex * 0.005)
    payload.extend(
        struct.pack(
            recordFormat,
            tUs,
            ax1,
            ay1,
            az1,
            gx1,
            gy1,
            gz1,
            ax2,
            ay2,
            az2,
            gx2,
            gy2,
            gz2,
            tempC,
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
timeSignals = signals_core.timeData(
    rawSignals,
    sampleRate,
    tempWindowSeconds,
)
freqSignals = signals_core.freqData(
    rawSignals,
    sampleRate,
    signals_core.fftConfig,
)
featureVector = ml_core.featureVector(timeSignals, freqSignals)

assert featureVector.shape == (18,)

print("packetRows:", len(rows))
print("featureShape:", featureVector.shape)
print("featureHead:", featureVector[:6])
print("offlineSmoke: PASS")
