################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import csv
import sys
import time

import serial

import data
import serial_shared

################################################################################
# variables/constants                                                          #
################################################################################
# Capture settings.
buildDir = Path(__file__).resolve().parents[1]
outRoot = buildDir / "data" / "training" / "main"
port = "/dev/cu.usbserial-0001"
baudRate = 1000000
timeout = 1.0
captureSec = 180.0
fileName = ""

################################################################################
# helpers                                                                      #
################################################################################


# Optional output filename.
def parseArgs():
    outFileName = fileName
    if len(sys.argv) > 1:
        outFileName = sys.argv[1]

    if outFileName == "":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        outFileName = f"capture_{stamp}.csv"
    if not outFileName.endswith(".csv"):
        outFileName += ".csv"

    return outFileName


# Captures one CSV recording.
def main():
    fileName = parseArgs()

    outRoot.mkdir(parents=True, exist_ok=True)
    savePath = outRoot / fileName

    print(f"starting capture of {savePath.name} for {captureSec} seconds")
    print(f"port: {port}")
    print(f"baudRate: {baudRate}")

    rowCount = 0
    leftOverBytes = b""
    firstTUsRaw = None
    startTime = 0.0

    with savePath.open("w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data.signalCols)

        with serial.Serial(port=port, baudrate=baudRate, timeout=timeout) as link:
            serial_shared.prepareLink(link)
            serial_shared.waitForReady(link, verbose=True)
            startTime = time.perf_counter()

            while (time.perf_counter() - startTime) < captureSec:
                newBytes = link.read(serial_shared.bufferSize)
                if len(newBytes) == 0:
                    continue

                leftOverBytes += newBytes
                packets, leftOverBytes = serial_shared.decodePackets(leftOverBytes)
                for packetValues in packets:
                    if firstTUsRaw is None:
                        firstTUsRaw = packetValues[0]
                    writer.writerow(
                        serial_shared.packetRow(packetValues, firstTUsRaw)
                    )
                    rowCount += 1

    runSecs = time.perf_counter() - startTime
    print("capture complete")
    print("rows logged:", rowCount)
    print("avg sample rate:", rowCount / runSecs)
    print("saved to:", savePath)


if __name__ == "__main__":
    main()
