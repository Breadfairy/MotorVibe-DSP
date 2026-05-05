################################################################################
# Imports                                                                      #
################################################################################
import math
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import serial

import charting_core
import signals_core

################################################################################
# variables/constants                                                          #
################################################################################
port = "/dev/cu.usbserial-0001"
baudRate = 1000000
timeout = 1.0

sampRate = 1000.0
plotSecs = 2.0
visSecs = 0.1
fftMinRows = 256

# Debug mode: read one 32-byte record at a time.
# Once streaming works, you can increase this back to 32.
bufferSampleCount = 0
recordFormat = "<I16hf"

accelScale = 16384.0
gyroScale = 131.0

readyPrefix = "Sample struct size (bytes):"

# Some firmware expects CRLF rather than only LF.
startCommand = b"START\r\n"

recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount

plotRows = int(sampRate * plotSecs)
visRows = max(1, int(sampRate * visSecs))
plotMaxHz = 500.0

################################################################################
# helpers                                                                      #
################################################################################


def resetDevice(link):
    # Toggles serial control lines to encourage ESP32 auto-reset.
    # Some boards ignore this, so manual reset still works if required.
    try:
        link.dtr = False
        link.rts = True
        time.sleep(0.1)

        link.dtr = True
        link.rts = False
        time.sleep(1.0)
    except OSError:
        pass


def waitForReady(link):
    # Waits for the firmware text handshake before binary streaming starts.
    print("waiting for device ready...")

    while True:
        deviceLine = link.readline().decode("ascii", errors="ignore").strip()

        if len(deviceLine) == 0:
            print("device:")
            continue

        print("device:", deviceLine)

        if deviceLine.startswith(readyPrefix):
            return


def startStream(link):
    # The ESP32 firmware is waiting for this before it sends binary samples.
    link.write(startCommand)
    link.flush()
    print(f"sent start command: {startCommand!r}")


def readPacket(link):
    # Reads one full binary packet. Returns None if nothing arrives.
    newBytes = link.read(bufferSize)

    if len(newBytes) == 0:
        print("no data received")
        return None

    if len(newBytes) != bufferSize:
        print(f"partial packet: {len(newBytes)} / {bufferSize} bytes")
        return None

    return newBytes


def decodeRows(newBytes):
    # Converts the binary packet into rows of usable sensor values.
    rows = []

    for offset in range(0, len(newBytes), recordSize):
        record = newBytes[offset:offset + recordSize]
        values = struct.unpack(recordFormat, record)

        tUs = values[0]
        raw = values[1:13]
        tempC = values[13]

        ax1 = raw[0] / accelScale
        ay1 = raw[1] / accelScale
        az1 = raw[2] / accelScale
        gx1 = raw[3] / gyroScale
        gy1 = raw[4] / gyroScale
        gz1 = raw[5] / gyroScale

        ax2 = raw[6] / accelScale
        ay2 = raw[7] / accelScale
        az2 = raw[8] / accelScale
        gx2 = raw[9] / gyroScale
        gy2 = raw[10] / gyroScale
        gz2 = raw[11] / gyroScale

        accMag1 = math.sqrt(ax1 * ax1 + ay1 * ay1 + az1 * az1)
        accMag2 = math.sqrt(ax2 * ax2 + ay2 * ay2 + az2 * az2)

        row = {
            "tUs": tUs,
            "tS": tUs / 1000000.0,
            "ax1": ax1,
            "ay1": ay1,
            "az1": az1,
            "gx1": gx1,
            "gy1": gy1,
            "gz1": gz1,
            "ax2": ax2,
            "ay2": ay2,
            "az2": az2,
            "gx2": gx2,
            "gy2": gy2,
            "gz2": gz2,
            "accMag1": accMag1,
            "accMag2": accMag2,
            "tempC": tempC,
        }

        rows.append(row)

    return rows


def buildLiveFig():
    plt.ion()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(charting_core.bgColor)

    charting_core.styleAx(axes[0])
    charting_core.styleAx(axes[1])

    axes[0].set_title("MPU6050 1 accMag live", color=charting_core.textColor)
    axes[0].set_xlabel("time s", color=charting_core.textColor)
    axes[0].set_ylabel("magnitude", color=charting_core.textColor)

    axes[1].set_title("MPU6050 1 acc fft", color=charting_core.textColor)
    axes[1].set_xlabel("frequency Hz", color=charting_core.textColor)
    axes[1].set_ylabel("magnitude", color=charting_core.textColor)
    axes[1].set_xlim(0.0, plotMaxHz)

    accLine, = axes[0].plot(
        [],
        [],
        color=charting_core.accColor,
        linewidth=1.2,
    )

    fftLine, = axes[1].plot(
        [],
        [],
        color=charting_core.fftColor,
        linewidth=1.2,
    )

    fig.tight_layout()

    return {
        "fig": fig,
        "axes": axes,
        "accLine": accLine,
        "fftLine": fftLine,
    }


def updateLiveFig(liveFig, rows):
    if len(rows) < 2:
        return

    rows = rows[-plotRows:]

    timeData = np.array([row["tS"] for row in rows], dtype=float)
    accData = np.array([row["accMag1"] for row in rows], dtype=float)

    timeData = timeData - timeData[0]

    liveFig["accLine"].set_data(timeData, accData)

    axTime = liveFig["axes"][0]
    axTime.set_xlim(0.0, max(plotSecs, timeData[-1]))

    accMin = float(np.min(accData))
    accMax = float(np.max(accData))
    accPad = max(0.05, (accMax - accMin) * 0.1)

    axTime.set_ylim(accMin - accPad, accMax + accPad)

    if len(accData) >= fftMinRows:
        fftData = accData - np.mean(accData)
        fftVals = np.abs(np.fft.rfft(fftData))
        fftFreq = np.fft.rfftfreq(len(fftData), d=1.0 / sampRate)

        liveFig["fftLine"].set_data(fftFreq, fftVals)

        axFft = liveFig["axes"][1]
        axFft.set_xlim(0.0, plotMaxHz)

        fftMax = float(np.max(fftVals))
        axFft.set_ylim(0.0, max(1.0, fftMax * 1.1))

    liveFig["fig"].canvas.draw_idle()
    plt.pause(0.001)


################################################################################
# main                                                                         #
################################################################################

print(f"starting visualiser on {port}")
print(f"baudRate: {baudRate}")
print(f"sampleRate: {sampRate}")
print(f"recordSize: {recordSize}")
print(f"bufferSampleCount: {bufferSampleCount}")
print(f"bufferSize: {bufferSize}")

liveFig = buildLiveFig()
plt.show(block=False)
plt.pause(0.1)

rows = []
newRows = 0

with serial.Serial(
    port=port,
    baudrate=baudRate,
    timeout=timeout,
) as link:
    resetDevice(link)
    link.reset_input_buffer()

    #waitForReady(link)

    time.sleep(0.1)
    startStream(link)

    while True:
        newBytes = readPacket(link)

        if newBytes is None:
            continue

        batchRows = decodeRows(newBytes)
        rows += batchRows
        newRows += len(batchRows)

        if len(rows) > plotRows:
            rows = rows[-plotRows:]

        if newRows >= visRows:
            updateLiveFig(liveFig, rows)
            newRows = 0
