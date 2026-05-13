################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import math
import os
import struct
import sys
import time

if "BUILD_V2_MPL_BACKEND" in os.environ:
    os.environ["MPLBACKEND"] = os.environ["BUILD_V2_MPL_BACKEND"]

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
buildDir = Path(__file__).resolve().parents[1]
modelPath = buildDir / "outputs" / "models" / "fanClassifier.joblib"
port = None
baudRate = 1000000
timeout = 1.0
recordFormat = "<I12hf"
bufferSampleCount = 32
accelScale = 16384.0
gyroScale = 131.0
readyPrefix = "Sample struct size (bytes):"
startCommand = b"START\n"
recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount
plotSecs = 2.0
visualRefreshSecs = 0.1
plotEventSecs = 0.25
inferRefreshSecs = 0.5
probSmoothSecs = 5.0
probHistorySecs = 60.0
plotMaxHz = 500.0
magSensor = 1
fftSensor = 1
bgColor = [0.08, 0.03, 0.03]
textColor = [0.98, 0.95, 0.82]
gridColor = [0.45, 0.40, 0.35]
magColor = "#f8f8f8"
fftColor = "#2a9d8f"
bpfoColor = "#d1495b"
bpfiColor = "#2f6fdd"
fundColor = "#f4a261"
classColors = [
    "#2f6fdd",
    "#d1495b",
    "#2a9d8f",
    "#f4a261",
    "#8e5ea2",
    "#6c757d",
]

################################################################################
# helpers                                                                      #
################################################################################


# Finds the first likely serial port if the user did not pass one in.
def detectPort():
    ports = list(list_ports.comports())
    for item in ports:
        if "usbserial" in item.device:
            return item.device
    for item in ports:
        if "usbmodem" in item.device:
            return item.device
    return None


# Waits for the MCU ready message and then starts the binary stream.
def waitForReady(link):
    while True:
        line = link.readline().decode("ascii", errors="ignore").strip()
        if len(line) == 0:
            continue
        if line.startswith(readyPrefix):
            link.write(startCommand)
            link.flush()
            return


# Resets the serial link so the MCU starts from a clean state.
def prepareLink(link):
    link.dtr = False
    link.rts = False
    time.sleep(0.2)
    link.reset_input_buffer()
    link.dtr = True
    link.rts = True


# Converts one decoded binary packet into one CSV-style signal row.
def packetRow(packetValues, firstTUsRaw):
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


# Applies the shared dark plot style to one axis.
def styleAx(ax):
    ax.set_facecolor(bgColor)
    ax.tick_params(colors=textColor, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(gridColor)
    ax.grid(
        color=gridColor,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )


# Returns the three acceleration column names for one MPU.
def accelCols(sensorNumber):
    return [
        f"ax{sensorNumber}",
        f"ay{sensorNumber}",
        f"az{sensorNumber}",
    ]


# Calculates acceleration magnitude for the live time plot.
def accelMagnitude(df, sensorNumber):
    cols = accelCols(sensorNumber)
    x = df[cols[0]].to_numpy(dtype=np.float64)
    y = df[cols[1]].to_numpy(dtype=np.float64)
    z = df[cols[2]].to_numpy(dtype=np.float64)
    return np.sqrt((x * x) + (y * y) + (z * z))


# Builds the summed acceleration FFT used by the live FFT plot.
def fftSum(df, sensorNumber, sampleRate):
    cols = accelCols(sensorNumber)
    sumParts = []
    hz = None
    for col in cols:
        hz, mag = signals.buildSpectrum(df[col], sampleRate)
        sumParts.append(mag)
    stacked = np.vstack(sumParts)
    total = np.sqrt(np.sum(stacked * stacked, axis=0))
    return hz, total


# Converts the latest live rows into one sklearn feature row.
def featureRowFromRows(rows, bundle):
    sampleRate = bundle["sampleRate"]
    df = pd.DataFrame(rows, columns=data.signalCols)
    df = data.cleanFrame(df)
    timeData = signals.timeSignals(df, sampleRate)
    freqData = signals.fftSignals(timeData, sampleRate, signals.fftConfig)
    return signals.modelInput(timeData, freqData)


# Converts probabilities into the top label and confidence value.
def stateFromProb(probVector, labelNames):
    topIndex = int(np.argmax(probVector))
    state = labelNames[topIndex]
    confidence = float(probVector[topIndex])
    return state, confidence


# Prints the current raw and smoothed state to the terminal.
def renderState(
    rawState,
    rawConfidence,
    rawProbVector,
    smoothState,
    smoothConfidence,
    smoothProbVector,
    labelNames,
):
    lines = [
        f"state: {smoothState}",
        f"confidence: {smoothConfidence * 100.0:.1f}%",
        "",
        "smoothed probabilities:",
    ]
    for index, labelName in enumerate(labelNames):
        lines.append(f"{labelName}: {smoothProbVector[index] * 100.0:.1f}%")

    lines += [
        "",
        "raw probabilities:",
        f"raw state: {rawState}",
        f"raw confidence: {rawConfidence * 100.0:.1f}%",
        "",
    ]
    for index, labelName in enumerate(labelNames):
        lines.append(f"{labelName}: {rawProbVector[index] * 100.0:.1f}%")
    text = "\n".join(lines)
    print(f"\x1b[2J\x1b[H{text}", end="", flush=True)


# Creates the acceleration magnitude subplot.
def buildMagnitudeFig(ax):
    ax.set_title(f"MPU{magSensor} Acc Magnitude", color=textColor)
    ax.set_xlabel("time s", color=textColor)
    ax.set_ylabel("magnitude", color=textColor)
    line, = ax.plot([], [], color=magColor, linewidth=1.2)
    return {
        "ax": ax,
        "line": line,
    }


# Creates the summed FFT subplot with bearing-band overlays.
def buildFftFig(ax):
    ax.set_title(f"MPU{fftSensor} Acc Summed FFT", color=textColor)
    ax.set_xlabel("frequency hz", color=textColor)
    ax.set_ylabel("magnitude", color=textColor)
    ax.set_xlim(0.0, plotMaxHz)
    bpfoRect = Rectangle(
        (0.0, 0.0),
        0.0,
        1.0,
        transform=ax.get_xaxis_transform(),
        color=bpfoColor,
        alpha=0.12,
        label="BPFO",
    )
    bpfiRect = Rectangle(
        (0.0, 0.0),
        0.0,
        1.0,
        transform=ax.get_xaxis_transform(),
        color=bpfiColor,
        alpha=0.10,
        label="BPFI",
    )
    ax.add_patch(bpfoRect)
    ax.add_patch(bpfiRect)
    line, = ax.plot([], [], color=fftColor, linewidth=1.2, label="FFT")
    fundLine = ax.axvline(
        0.0,
        color=fundColor,
        linewidth=1.0,
        alpha=0.8,
        label="Fund",
    )
    fundDot, = ax.plot(
        [],
        [],
        color=fundColor,
        marker="o",
        linestyle="None",
        markersize=4,
        zorder=3,
    )
    ax.legend(frameon=False, labelcolor=textColor, loc="upper right")
    return {
        "ax": ax,
        "line": line,
        "bpfoRect": bpfoRect,
        "bpfiRect": bpfiRect,
        "fundLine": fundLine,
        "fundDot": fundDot,
    }


# Creates the raw probability trace subplot.
def buildProbFig(ax, labelNames):
    ax.set_title("Raw Model Probability", color=textColor)
    ax.set_xlabel("elapsed s", color=textColor)
    ax.set_ylabel("probability %", color=textColor)
    ax.set_ylim(0.0, 100.0)
    lines = []
    for index, labelName in enumerate(labelNames):
        line, = ax.plot(
            [],
            [],
            color=classColors[index],
            linewidth=1.2,
            label=labelName,
        )
        lines.append(line)
    ax.legend(frameon=False, labelcolor=textColor, loc="upper left")
    return {
        "ax": ax,
        "lines": lines,
    }


# Creates the single live figure with all three subplots.
def buildLiveFigs(labelNames):
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    fig.patch.set_facecolor(bgColor)
    for ax in axes:
        styleAx(ax)

    figs = {
        "fig": fig,
        "mag": buildMagnitudeFig(axes[0]),
        "fft": buildFftFig(axes[1]),
        "prob": buildProbFig(axes[2], labelNames),
    }
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return figs


# Updates the acceleration magnitude line.
def updateMagnitudeFig(figData, df):
    t = df["t_s"].to_numpy(dtype=np.float64)
    y = accelMagnitude(df, magSensor)
    t = t - t[0]
    figData["line"].set_data(t, y)
    figData["ax"].set_xlim(0.0, max(plotSecs, float(t[-1])))
    yMin = float(np.min(y))
    yMax = float(np.max(y))
    yPad = max(0.05, (yMax - yMin) * 0.1)
    figData["ax"].set_ylim(yMin - yPad, yMax + yPad)


# Updates the FFT line, fundamental marker, and BPFO/BPFI bands.
def updateFftFig(figData, df, sampleRate):
    hz, total = fftSum(df, fftSensor, sampleRate)
    fundHz, fundMag = signals.fundamentalPeak(
        hz,
        total,
        signals.fftConfig["minHz"],
        signals.fftConfig["maxHz"],
    )
    bpfo = signals.orderBand(
        fundHz,
        signals.fftConfig["bpfoLowOrder"],
        signals.fftConfig["bpfoHighOrder"],
        signals.fftConfig["tolFraction"],
    )
    bpfi = signals.orderBand(
        fundHz,
        signals.fftConfig["bpfiLowOrder"],
        signals.fftConfig["bpfiHighOrder"],
        signals.fftConfig["tolFraction"],
    )
    mask = hz <= plotMaxHz
    x = hz[mask]
    y = total[mask]
    figData["line"].set_data(x, y)
    figData["bpfoRect"].set_x(bpfo[0])
    figData["bpfoRect"].set_width(bpfo[1] - bpfo[0])
    figData["bpfiRect"].set_x(bpfi[0])
    figData["bpfiRect"].set_width(bpfi[1] - bpfi[0])
    figData["fundLine"].set_xdata([fundHz, fundHz])
    figData["fundDot"].set_data([fundHz], [fundMag])
    figData["ax"].set_title(
        f"MPU{fftSensor} Acc Summed FFT | fundamental {fundHz:.1f} Hz",
        color=textColor,
    )
    yMax = float(np.max(y))
    figData["ax"].set_ylim(0.0, max(0.1, yMax * 1.1))


# Updates the raw class probability lines.
def updateProbFig(figData, probTimes, probRows, labelNames):
    x = np.array(probTimes, dtype=np.float64)
    y = np.vstack(probRows).astype(np.float64) * 100.0
    for index in range(len(labelNames)):
        figData["lines"][index].set_data(x, y[:, index])
    if len(x) > 1:
        figData["ax"].set_xlim(max(0.0, x[-1] - probHistorySecs), x[-1])
    else:
        figData["ax"].set_xlim(0.0, probHistorySecs)


# Updates the fast visual plots from the newest live rows.
def updateVisualFigs(figs, rows, sampleRate, visualRows):
    viewRows = rows[-visualRows:]
    df = pd.DataFrame(viewRows, columns=data.signalCols)
    updateMagnitudeFig(figs["mag"], df)
    updateFftFig(figs["fft"], df, sampleRate)


# Gives matplotlib time to redraw without blocking every loop pass.
def servicePlot(fig):
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


# Stores one probability row and trims old history.
def appendProbHistory(probTimes, probRows, elapsedSecs, probVector):
    probTimes.append(elapsedSecs)
    probRows.append(probVector)
    while len(probTimes) > 0 and probTimes[0] < elapsedSecs - probHistorySecs:
        del probTimes[0]
        del probRows[0]


# Averages recent raw probabilities for the terminal state readout.
def smoothProbVector(rawProbTimes, rawProbRows, elapsedSecs):
    activeRows = []
    for index in range(len(rawProbTimes)):
        if rawProbTimes[index] >= elapsedSecs - probSmoothSecs:
            activeRows.append(rawProbRows[index])

    if len(activeRows) == 0:
        return rawProbRows[-1]

    probMatrix = np.vstack(activeRows).astype(np.float64)
    smoothVector = np.mean(probMatrix, axis=0)
    total = float(np.sum(smoothVector))
    if total > 0.0:
        smoothVector = smoothVector / total
    return smoothVector

################################################################################
# main functions                                                               #
################################################################################


# Runs the live serial, plotting, and inference loop.
def main():
    global port
    if len(sys.argv) > 1:
        port = sys.argv[1]
    if port is None:
        port = detectPort()
    if port is None:
        raise SystemExit(
            "No USB serial port found. Pass the port explicitly, for example: "
            "python3 src/live.py /dev/cu.usbserial-0001"
        )

    bundle = joblib.load(modelPath)
    modelValue = bundle["model"]
    labelNames = bundle["labelNames"]
    winRows = int(bundle["sampleRate"] * bundle["winSecs"])
    visualRows = int(bundle["sampleRate"] * plotSecs)
    keepRows = max(winRows, visualRows)
    liveFigs = buildLiveFigs(labelNames)

    rows = []
    remBytes = b""
    firstTUsRaw = None
    startTime = 0.0
    lastVisualUpdate = 0.0
    lastInferUpdate = 0.0
    lastPlotEvent = 0.0
    plotDirty = False
    probTimes = []
    probRows = []
    rawProbTimes = []
    rawProbRows = []

    with serial.Serial(
        port=port,
        baudrate=baudRate,
        timeout=timeout,
    ) as link:
        prepareLink(link)
        waitForReady(link)
        startTime = time.perf_counter()
        lastVisualUpdate = startTime
        lastInferUpdate = startTime
        lastPlotEvent = startTime

        while True:
            newBytes = link.read(bufferSize)
            if len(newBytes) == 0:
                continue

            remBytes += newBytes
            while len(remBytes) >= recordSize:
                packetBytes = remBytes[:recordSize]
                remBytes = remBytes[recordSize:]
                packetValues = struct.unpack(recordFormat, packetBytes)
                if firstTUsRaw is None:
                    firstTUsRaw = packetValues[0]
                rows.append(packetRow(packetValues, firstTUsRaw))

            if len(rows) > keepRows:
                rows = rows[-keepRows:]
            if len(rows) < min(winRows, visualRows):
                continue

            now = time.perf_counter()
            if now - lastVisualUpdate >= visualRefreshSecs:
                lastVisualUpdate = now
                updateVisualFigs(
                    liveFigs,
                    rows,
                    bundle["sampleRate"],
                    visualRows,
                )
                plotDirty = True

            if len(rows) < winRows:
                if plotDirty and now - lastPlotEvent >= plotEventSecs:
                    servicePlot(liveFigs["fig"])
                    plotDirty = False
                    lastPlotEvent = now
                continue
            if now - lastInferUpdate < inferRefreshSecs:
                if plotDirty and now - lastPlotEvent >= plotEventSecs:
                    servicePlot(liveFigs["fig"])
                    plotDirty = False
                    lastPlotEvent = now
                continue
            lastInferUpdate = now

            featureRow = featureRowFromRows(rows[-winRows:], bundle)
            rawProbVector = modelValue.predict_proba([featureRow])[0]
            elapsedSecs = now - startTime
            appendProbHistory(
                rawProbTimes,
                rawProbRows,
                elapsedSecs,
                rawProbVector,
            )
            probVector = smoothProbVector(
                rawProbTimes,
                rawProbRows,
                elapsedSecs,
            )
            rawState, rawConfidence = stateFromProb(rawProbVector, labelNames)
            state, confidence = stateFromProb(probVector, labelNames)
            renderState(
                rawState,
                rawConfidence,
                rawProbVector,
                state,
                confidence,
                probVector,
                labelNames,
            )
            appendProbHistory(probTimes, probRows, elapsedSecs, rawProbVector)
            updateProbFig(liveFigs["prob"], probTimes, probRows, labelNames)
            plotDirty = True

            if plotDirty and now - lastPlotEvent >= plotEventSecs:
                servicePlot(liveFigs["fig"])
                plotDirty = False
                lastPlotEvent = now


if __name__ == "__main__":
    main()
