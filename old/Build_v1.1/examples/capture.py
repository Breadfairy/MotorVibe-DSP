# ################################ Capture.py  #################################
import csv
import os
import serial
import struct
import sys
import time

# ######################## Configurable Variables ##############################

captureTime = 10
fileName = "test1.csv"
serialPort = "COM5"
baudRate = 921600
serialTimeout = 1.0
recordFormat = "<I6h"
bufferSampleCount = 32

recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount

csvColumns = [
    "t_us",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]

# ############################ Script Body ####################################

if len(sys.argv) > 1:
    fileName = sys.argv[1]
    if not fileName.endswith(".csv"):
        fileName += ".csv"

os.makedirs("ML_data", exist_ok=True)

savePath = "ML_data/" + fileName
rowsLogged = 0
startTime = time.perf_counter()
leftOverBytes = b""

print(
    f"starting capture of {fileName} "
    f"for {captureTime} seconds"
)

with open(savePath, "w", newline="") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(csvColumns)

    with serial.Serial(
        port=serialPort,
        baudrate=baudRate,
        timeout=serialTimeout,
    ) as link:
        link.reset_input_buffer()

        while (time.perf_counter() - startTime) < captureTime:
            newBytes = link.read(bufferSize)
            if len(newBytes) == 0:
                continue

            leftOverBytes += newBytes

            while len(leftOverBytes) >= recordSize:
                packet = leftOverBytes[:recordSize]
                leftOverBytes = leftOverBytes[recordSize:]
                row = struct.unpack(recordFormat, packet)
                writer.writerow(row)
                rowsLogged += 1

print("capture complete")
print("rows logged:", rowsLogged)
print("saved to:", savePath)
