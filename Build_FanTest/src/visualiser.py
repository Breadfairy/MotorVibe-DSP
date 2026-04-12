################################################################################
# Imports                                                                      #
################################################################################
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
port = "COM5"
baudRate = 1000000
timeout = 1.0
sampRate = 1000.0
plotSecs = 2.0
visSecs = 0.1
fftMinRows = 256
bufferSampleCount = 32
recordFormat = "<I6h"
accelScale = 16384.0
gyroScale = 131.0
readyPrefix = "Sample struct size (bytes):"
startCommand = b"START\n"
recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount
plotRows = int(sampRate * plotSecs)
plotMaxHz = 500.0

################################################################################
# helpers                                                                      #
################################################################################


# Builds one live 1x2 figure for sensor 1 magnitude and FFT monitoring.
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
    axes[1].set_xlabel("frequency hz", color=charting_core.textColor)
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
        linewidth=1.0,
    )
    fundMark = axes[1].scatter(
        [],
        [],
        color=charting_core.bpfoColor,
        s=28,
        zorder=3,
    )
    bpfoPatch = axes[1].axvspan(
        0.0,
        0.0,
        color=charting_core.bpfoColor,
        alpha=0.12,
    )
    bpfiPatch = axes[1].axvspan(
        0.0,
        0.0,
        color=charting_core.bpfiColor,
        alpha=0.10,
    )
    liveFig = {
        "fig": fig,
        "accAx": axes[0],
        "fftAx": axes[1],
        "accLine": accLine,
        "fftLine": fftLine,
        "fundMark": fundMark,
        "bpfoPatch": bpfoPatch,
        "bpfiPatch": bpfiPatch,
    }
    fig.tight_layout(pad=0.8)
    plt.show(block=False)
    return liveFig


# Builds the live visual signals from the current rolling rows.
def buildVisData(rows, sampRate):
    rowData = np.asarray(rows, dtype=np.float64)
    timeAxis = np.arange(rowData.shape[0], dtype=np.float64) / sampRate
    axisX = rowData[:, 2]
    axisY = rowData[:, 3]
    axisZ = rowData[:, 4]
    accMag = signals_core.mag3(axisX, axisY, axisZ)
    freqAxis, xSpec = signals_core.buildSpectrum(axisX, sampRate)
    _, ySpec = signals_core.buildSpectrum(axisY, sampRate)
    _, zSpec = signals_core.buildSpectrum(axisZ, sampRate)
    accSpec = np.sqrt(xSpec**2 + ySpec**2 + zSpec**2)
    fundHz, fundMag = signals_core.fundamentalPeak(
        freqAxis,
        accSpec,
        signals_core.fftConfig["minHz"],
        signals_core.fftConfig["maxHz"],
    )
    bpfoBand = signals_core.orderBand(
        fundHz,
        signals_core.fftConfig["bpfoLowOrder"],
        signals_core.fftConfig["bpfoHighOrder"],
        signals_core.fftConfig["tolFraction"],
    )
    bpfiBand = signals_core.orderBand(
        fundHz,
        signals_core.fftConfig["bpfiLowOrder"],
        signals_core.fftConfig["bpfiHighOrder"],
        signals_core.fftConfig["tolFraction"],
    )
    visData = {
        "timeAxis": timeAxis,
        "accMag": accMag,
        "freqAxis": freqAxis,
        "accSpec": accSpec,
        "fundHz": fundHz,
        "fundMag": fundMag,
        "bpfoBand": bpfoBand,
        "bpfiBand": bpfiBand,
    }
    return visData


# Updates the live figure from the latest visual-only signal data.
def updateLiveFig(liveFig, visData):
    accAx = liveFig["accAx"]
    fftAx = liveFig["fftAx"]
    accLine = liveFig["accLine"]
    fftLine = liveFig["fftLine"]
    fundMark = liveFig["fundMark"]
    plotMask = visData["freqAxis"] <= plotMaxHz
    plotFreq = visData["freqAxis"][plotMask]
    plotSpec = visData["accSpec"][plotMask]

    accLine.set_data(visData["timeAxis"], visData["accMag"])
    accAx.set_xlim(visData["timeAxis"][0], visData["timeAxis"][-1])
    accMin = float(np.min(visData["accMag"]))
    accMax = float(np.max(visData["accMag"]))
    if accMin == accMax:
        accMax = accMin + 0.1
    accAx.set_ylim(accMin, accMax)

    fftLine.set_data(plotFreq, plotSpec)
    fftAx.set_xlim(0.0, plotMaxHz)
    fftMax = float(np.max(plotSpec))
    if fftMax <= 0.0:
        fftMax = 0.1
    fftAx.set_ylim(0.0, fftMax * 1.1)
    fundMark.set_offsets([[visData["fundHz"], visData["fundMag"]]])
    liveFig["bpfoPatch"].remove()
    liveFig["bpfiPatch"].remove()
    liveFig["bpfoPatch"] = fftAx.axvspan(
        visData["bpfoBand"][0],
        visData["bpfoBand"][1],
        color=charting_core.bpfoColor,
        alpha=0.12,
    )
    liveFig["bpfiPatch"] = fftAx.axvspan(
        visData["bpfiBand"][0],
        visData["bpfiBand"][1],
        color=charting_core.bpfiColor,
        alpha=0.10,
    )
    fftAx.set_title(
        f"MPU6050 1 acc fft ({visData['fundHz']:.2f} hz)",
        color=charting_core.textColor,
    )
    liveFig["fig"].canvas.draw_idle()
    liveFig["fig"].canvas.flush_events()
    plt.pause(0.001)


# Waits for the firmware text handshake before binary streaming starts.
def waitForReady(link):
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


rows = []
remBytes = b""
visTs = time.perf_counter()
firstTUsRaw = None

print(f"starting visualiser on {port}")
print(f"baudRate: {baudRate}")
print(f"sampleRate: {sampRate}")
print("waiting for device ready...")

################################################################################
# live visual setup                                                            #
################################################################################

liveFig = buildLiveFig()

with serial.Serial(
    port=port,
    baudrate=baudRate,
    timeout=timeout,
) as link:
    waitForReady(link)

    while True:
        newBytes = link.read(bufferSize)
        if len(newBytes) == 0:
            continue

        remBytes += newBytes

        while len(remBytes) >= recordSize:
            packetBytes = remBytes[:recordSize]
            remBytes = remBytes[recordSize:]
            packetValues = struct.unpack(recordFormat, packetBytes)
            tUsRaw = packetValues[0]
            if firstTUsRaw is None:
                firstTUsRaw = tUsRaw
            tUs = (tUsRaw - firstTUsRaw) & 0xFFFFFFFF
            rowValue = [
                tUs,
                tUs / 1e6,
                packetValues[1] / accelScale,
                packetValues[2] / accelScale,
                packetValues[3] / accelScale,
                packetValues[4] / gyroScale,
                packetValues[5] / gyroScale,
                packetValues[6] / gyroScale,
            ]
            rows.append(rowValue)

        if len(rows) > plotRows:
            rows = rows[-plotRows:]

        if len(rows) < fftMinRows:
            continue

        nowTs = time.perf_counter()
        if nowTs - visTs < visSecs:
            continue

        visTs = nowTs

        ########################################################################
        # live visual update                                                   #
        ########################################################################

        visData = buildVisData(rows, sampRate)
        updateLiveFig(liveFig, visData)
