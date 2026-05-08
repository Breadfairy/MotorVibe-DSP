from pathlib import Path
import csv
import math
import struct
import time

import serial


captureSec = 10.0
port = "/dev/cu.usbserial-0001"
baudRate = 1000000
timeout = 1.0
bufferSampleCount = 32
recordFormat = "<I12hf"
accelScale = 16384.0
gyroScale = 131.0
readyPrefix = "Sample struct size (bytes):"
startCommand = b"START\n"
recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount
savePath = Path("BuildBACKUP/python/ML_data/codex_backup_wiggle_clean.csv")

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


def waitForReady(link):
    print("waiting for device ready...")
    startTime = time.perf_counter()
    while time.perf_counter() - startTime < 8.0:
        deviceLine = link.readline().decode("ascii", errors="ignore").strip()
        if len(deviceLine) == 0:
            continue
        print("device:", deviceLine)
        if deviceLine.startswith(readyPrefix):
            link.write(startCommand)
            link.flush()
            return True
    return False


def writeRow(writer, packetValues, firstTUsRaw):
    tUsRaw = packetValues[0]
    tUs = (tUsRaw - firstTUsRaw) & 0xFFFFFFFF
    tempC = packetValues[13]
    if not math.isfinite(tempC):
        tempC = 0.0

    writer.writerow([
        tUs,
        tUs / 1000000.0,
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
    ])


savePath.parent.mkdir(parents=True, exist_ok=True)
rowCount = 0
leftOverBytes = b""
firstTUsRaw = None

print("backup clean capture")
print("file:", savePath)
print("captureSec:", captureSec)
print("baudRate:", baudRate)

with serial.Serial(port=port, baudrate=baudRate, timeout=timeout) as link:
    link.dtr = False
    link.rts = False
    time.sleep(0.2)
    link.reset_input_buffer()
    link.dtr = True
    link.rts = True

    if not waitForReady(link):
        raise SystemExit("device ready timeout")

    startTime = time.perf_counter()
    with savePath.open("w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(csvColumns)

        while time.perf_counter() - startTime < captureSec:
            newBytes = link.read(bufferSize)
            if len(newBytes) == 0:
                continue

            leftOverBytes += newBytes
            while len(leftOverBytes) >= recordSize:
                packetBytes = leftOverBytes[:recordSize]
                leftOverBytes = leftOverBytes[recordSize:]
                packetValues = struct.unpack(recordFormat, packetBytes)

                if firstTUsRaw is None:
                    firstTUsRaw = packetValues[0]
                writeRow(writer, packetValues, firstTUsRaw)
                rowCount += 1

runSecs = time.perf_counter() - startTime
print("capture complete")
print("rows logged:", rowCount)
print("avg sample rate:", rowCount / runSecs)
print("saved to:", savePath)
