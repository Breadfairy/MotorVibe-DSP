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
buildDir = Path(__file__).resolve().parents[1]
outRoot = buildDir / "data" / "training" / "main"
defaultPort = "/dev/cu.usbserial-0001"
baudRate = 1000000
timeout = 1.0
defaultSeconds = 10.0
defaultLabel = "good"
bufferSampleCount = 32
recordFormat = "<I12hf"
accelScale = 16384.0
gyroScale = 131.0
readyPrefix = "Sample struct size (bytes):"
startCommand = b"START\n"
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
validLabels = ["good", "voltSag", "obstruction", "imbalance"]

################################################################################
# helpers                                                                      #
################################################################################


# Waits for the MCU ready message and starts streaming.
def waitForReady(link):
    print("waiting for device ready...")
    while True:
        deviceLine = link.readline().decode("ascii", errors="ignore").strip()
        if len(deviceLine) == 0:
            continue
        print("device:", deviceLine)
        if deviceLine.startswith(readyPrefix):
            link.write(startCommand)
            link.flush()
            return


# Resets the serial link before capture starts.
def prepareLink(link):
    link.dtr = False
    link.rts = False
    time.sleep(0.2)
    link.reset_input_buffer()
    link.dtr = True
    link.rts = True


# Converts one binary packet into one CSV row.
def packetRow(packetValues, firstTUsRaw):
    tUsRaw = packetValues[0]
    tUs = (tUsRaw - firstTUsRaw) & 0xFFFFFFFF
    tempC = packetValues[13]
    if not math.isfinite(tempC):
        tempC = 0.0

    return [
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
    ]

################################################################################
# main functions                                                               #
################################################################################


# Captures one labelled recording into the training data folder.
def main():
    label = defaultLabel
    fileName = ""
    seconds = defaultSeconds
    port = defaultPort

    if len(sys.argv) > 1:
        label = sys.argv[1]
    if len(sys.argv) > 2:
        fileName = sys.argv[2]
    if len(sys.argv) > 3:
        seconds = float(sys.argv[3])
    if len(sys.argv) > 4:
        port = sys.argv[4]

    if label not in validLabels:
        raise SystemExit(f"label must be one of: {', '.join(validLabels)}")

    if fileName == "":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fileName = f"{label}_{stamp}.csv"
    if not fileName.endswith(".csv"):
        fileName += ".csv"

    outDir = outRoot / label
    outDir.mkdir(parents=True, exist_ok=True)
    savePath = outDir / fileName

    print(f"starting capture of {savePath.name} for {seconds} seconds")
    print(f"label: {label}")
    print(f"port: {port}")
    print(f"baudRate: {baudRate}")

    rowCount = 0
    leftOverBytes = b""
    firstTUsRaw = None
    startTime = 0.0

    with savePath.open("w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(csvColumns)

        with serial.Serial(port=port, baudrate=baudRate, timeout=timeout) as link:
            prepareLink(link)
            waitForReady(link)
            startTime = time.perf_counter()

            while (time.perf_counter() - startTime) < seconds:
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
                    writer.writerow(packetRow(packetValues, firstTUsRaw))
                    rowCount += 1

    runSecs = time.perf_counter() - startTime
    print("capture complete")
    print("rows logged:", rowCount)
    print("avg sample rate:", rowCount / runSecs)
    print("saved to:", savePath)


if __name__ == "__main__":
    main()
