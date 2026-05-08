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
# Capture writes raw labelled recordings into the training folder.
# These paths and serial settings are the main values to change before a lab run.
buildDir = Path(__file__).resolve().parents[1]
outRoot = buildDir / "data" / "training" / "main"
port = "/dev/cu.usbserial-0001"
baudRate = 1000000
timeout = 1.0
captureSec = 180.0
labelName = "good"
fileName = ""

# The firmware sends a fixed binary packet:
# uint32 timestamp, 12 signed int16 sensor readings, and one float temperature.
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

# These are the only class names accepted by the rest of the ML code.
# The filename is forced to start with one of these labels.
validLabels = [
    "good",
    "bad_leak",
]

################################################################################
# helpers                                                                      #
################################################################################


# Reads the label, file name, capture time, and optional port.
# Defaults are set above so a simple run captures a good recording.
# Command line values only override those basic variables.
def parseArgs():
    global labelName
    global fileName
    global captureSec
    global port

    if len(sys.argv) > 1:
        labelName = sys.argv[1]
    if len(sys.argv) > 2:
        fileName = sys.argv[2]
    if len(sys.argv) > 3:
        captureSec = float(sys.argv[3])
    if len(sys.argv) > 4:
        port = sys.argv[4]

    if labelName not in validLabels:
        raise SystemExit("label must be good or bad_leak")

    # If no file name is given, make one from the label and timestamp.
    # If a file name is given without the label prefix, add it automatically.
    if fileName == "":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fileName = f"{labelName}_{stamp}.csv"
    if not fileName.endswith(".csv"):
        fileName += ".csv"
    if not fileName.startswith(labelName):
        fileName = f"{labelName}_{fileName}"


# Waits for the MCU ready message and starts streaming.
# The MCU prints a ready line after boot/reset. Once that appears, this sends
# START so both capture.py and live.py begin from a known point.
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
# Toggling DTR/RTS gives the board a clean serial start and clears old bytes.
def prepareLink(link):
    link.dtr = False
    link.rts = False
    time.sleep(0.2)
    link.reset_input_buffer()
    link.dtr = True
    link.rts = True


# Converts one binary packet into one CSV row.
# This is where raw integer sensor counts become real units:
# accel in g, gyro in deg/s, and time in seconds.
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
    parseArgs()

    # Create the output folder and open the CSV before starting the device.
    outRoot.mkdir(parents=True, exist_ok=True)
    savePath = outRoot / fileName

    print(f"starting capture of {savePath.name} for {captureSec} seconds")
    print(f"label: {labelName}")
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

            # Read bytes until captureSec has elapsed.
            # Bytes may arrive in chunks that do not line up exactly with packet
            # boundaries, so leftOverBytes keeps any partial packet for later.
            while (time.perf_counter() - startTime) < captureSec:
                newBytes = link.read(bufferSize)
                if len(newBytes) == 0:
                    continue

                leftOverBytes += newBytes
                while len(leftOverBytes) >= recordSize:
                    # Pull one full packet out of the byte buffer and write it
                    # as one row in the CSV file.
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
