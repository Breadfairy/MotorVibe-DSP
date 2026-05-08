from pathlib import Path
import csv
import math
import struct
import time

import serial


captureSec = 10.0
port = "/dev/cu.usbserial-0001"
baudRate = 1000000
recFmt = "<I12hf"
recSize = struct.calcsize(recFmt)
bufferSampleCount = 32
bufferSize = recSize * bufferSampleCount
readyPrefix = "Sample struct size (bytes):"
startCommand = b"START\n"
accelScale = 16384.0
gyroScale = 131.0
savePath = Path("Build_v2/data/clean/codex_original_firmware_capture.csv")

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
    startWait = time.perf_counter()
    while time.perf_counter() - startWait < 8.0:
        deviceLine = link.readline().decode("ascii", errors="ignore").strip()
        if len(deviceLine) == 0:
            continue
        print("device:", deviceLine)
        if deviceLine.startswith(readyPrefix):
            link.write(startCommand)
            link.flush()
            return True
    return False


def scaledRow(packetValues, firstTUsRaw):
    tUsRaw = packetValues[0]
    tUs = (tUsRaw - firstTUsRaw) & 0xFFFFFFFF
    tempC = packetValues[13]
    if not math.isfinite(tempC):
        tempC = 0.0
    return [
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


savePath.parent.mkdir(parents=True, exist_ok=True)

rowCount = 0
leftOverBytes = b""
firstTUsRaw = None
lastTUs = None
dtMin = None
dtMax = None
dtTotal = 0
dtCount = 0
sameCount = 0
diffCount = 0

print("original firmware capture test")
print("port:", port)
print("baudRate:", baudRate)
print("captureSec:", captureSec)

with serial.Serial(port=port, baudrate=baudRate, timeout=1.0) as link:
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
            while len(leftOverBytes) >= recSize:
                packetBytes = leftOverBytes[:recSize]
                leftOverBytes = leftOverBytes[recSize:]
                packetValues = struct.unpack(recFmt, packetBytes)

                tUsRaw = packetValues[0]
                if firstTUsRaw is None:
                    firstTUsRaw = tUsRaw
                tUs = (tUsRaw - firstTUsRaw) & 0xFFFFFFFF

                if lastTUs is not None:
                    dt = tUs - lastTUs
                    if dt >= 0:
                        dtMin = dt if dtMin is None else min(dtMin, dt)
                        dtMax = dt if dtMax is None else max(dtMax, dt)
                        dtTotal += dt
                        dtCount += 1
                lastTUs = tUs

                firstValues = packetValues[1:7]
                secondValues = packetValues[7:13]
                if firstValues == secondValues:
                    sameCount += 1
                else:
                    diffCount += 1

                writer.writerow(scaledRow(packetValues, firstTUsRaw))
                rowCount += 1

runSecs = time.perf_counter() - startTime
timestampRate = 0.0
if lastTUs is not None and lastTUs > 0:
    timestampRate = rowCount / (lastTUs / 1000000.0)

dtMean = 0.0
if dtCount > 0:
    dtMean = dtTotal / dtCount

print("capture complete")
print("rows:", rowCount)
print("host_rate:", rowCount / runSecs)
print("timestamp_rate:", timestampRate)
print("timestamp_span_s:", 0.0 if lastTUs is None else lastTUs / 1000000.0)
print("dt_us_min:", dtMin)
print("dt_us_mean:", dtMean)
print("dt_us_max:", dtMax)
print("same_sensor_rows:", sameCount)
print("different_sensor_rows:", diffCount)
print("saved:", savePath)
