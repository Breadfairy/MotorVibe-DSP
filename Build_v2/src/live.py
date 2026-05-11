################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import serial

import data
import ml
import serial_shared
import signals

################################################################################
# variables/constants                                                          #
################################################################################
# Live serial and model settings.
buildDir = Path(__file__).resolve().parents[1]
modelPath = buildDir / "outputs" / "models" / "motorPumpClassifier.joblib"
port = "/dev/cu.usbserial-0001"
baudRate = 1000000
timeout = 1.0

# Plot and inference timing.
plotSecs = 2.0
visualRefreshSecs = 0.1
plotEventSecs = 0.25
inferRefreshSecs = 0.5
probSmoothSecs = 5.0
probHistorySecs = 60.0
plotMaxHz = 500.0
magSensor = 1
fftSensor = 1

# Display colours.
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


# Applies the chart style.
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


# Prints the current state in place.
def renderState(state, confidence, probVector, labelNames):
    lines = [
        f"state: {state}",
        f"confidence: {confidence * 100.0:.1f}%",
        "",
    ]
    for index, labelName in enumerate(labelNames):
        lines.append(f"{labelName}: {probVector[index] * 100.0:.1f}%")
    text = "\n".join(lines)
    print(f"\x1b[2J\x1b[H{text}", end="", flush=True)


# Builds the acceleration magnitude plot.
def buildMagnitudeFig(ax):
    ax.set_title(f"MPU{magSensor} Acc Magnitude", color=textColor)
    ax.set_xlabel("time s", color=textColor)
    ax.set_ylabel("magnitude", color=textColor)
    line, = ax.plot([], [], color=magColor, linewidth=1.2)
    return {
        "ax": ax,
        "line": line,
    }


# Builds the FFT plot.
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


# Builds the probability history plot.
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


# Creates the live figure.
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


# Updates the time-domain plot.
def updateMagnitudeFig(figData, df, sampleRate):
    cols = signals.accelCols(magSensor)
    timeData = signals.plotTime(df, sampleRate, columns=cols)
    x = timeData[cols[0]]
    yAxis = timeData[cols[1]]
    z = timeData[cols[2]]
    y = np.sqrt((x * x) + (yAxis * yAxis) + (z * z))
    t = df["t_s"].to_numpy(dtype=np.float64)
    t = t - t[0]
    figData["line"].set_data(t, y)
    figData["ax"].set_xlim(0.0, max(plotSecs, float(t[-1])))
    yMin = float(np.min(y))
    yMax = float(np.max(y))
    yPad = max(0.05, (yMax - yMin) * 0.1)
    figData["ax"].set_ylim(yMin - yPad, yMax + yPad)


# Updates the FFT plot.
def updateFftFig(figData, df, sampleRate):
    cols = signals.accelCols(fftSensor)
    timeData = signals.plotTime(df, sampleRate, columns=cols)
    freqData = signals.plotFreq(timeData, sampleRate, signals.fftConfig)
    hz = freqData["freqAxis"]
    parts = np.vstack([freqData[f"{col}Spectrum"] for col in cols])
    total = np.sqrt(np.sum(parts * parts, axis=0))
    fundHz, fundMag = signals.fundamentalPeak(
        hz,
        total,
        signals.fftConfig["minHz"],
        signals.fftConfig["maxHz"],
    )
    bpfo, bpfi = signals.bearingBands(fundHz, signals.fftConfig)
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


# Updates the probability plot.
def updateProbFig(figData, probTimes, probRows, labelNames):
    x = np.array(probTimes, dtype=np.float64)
    y = np.vstack(probRows).astype(np.float64) * 100.0
    for index in range(len(labelNames)):
        figData["lines"][index].set_data(x, y[:, index])
    if len(x) > 1:
        figData["ax"].set_xlim(max(0.0, x[-1] - probHistorySecs), x[-1])
    else:
        figData["ax"].set_xlim(0.0, probHistorySecs)


# Updates the signal plots.
def updateVisualFigs(figs, rows, sampleRate, visualRows):
    viewRows = rows[-visualRows:]
    df = pd.DataFrame(viewRows, columns=data.signalCols)
    updateMagnitudeFig(figs["mag"], df, sampleRate)
    updateFftFig(figs["fft"], df, sampleRate)


# Services matplotlib events.
def servicePlot(fig):
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


# Stores probability history.
def appendProbHistory(probTimes, probRows, elapsedSecs, probVector):
    probTimes.append(elapsedSecs)
    probRows.append(probVector)
    while len(probTimes) > 0 and probTimes[0] < elapsedSecs - probHistorySecs:
        del probTimes[0]
        del probRows[0]


# Smooths the displayed state.
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


def main():
    classifier = ml.MotorPumpClassifier.load(modelPath)
    labelNames = classifier.labelNames
    winRows = int(classifier.sampleRate * classifier.winSecs)
    visualRows = int(classifier.sampleRate * plotSecs)
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

    with serial.Serial(
        port=port,
        baudrate=baudRate,
        timeout=timeout,
    ) as link:
        serial_shared.prepareLink(link)
        serial_shared.waitForReady(link)
        startTime = time.perf_counter()
        lastVisualUpdate = startTime
        lastInferUpdate = startTime
        lastPlotEvent = startTime

        while True:
            newBytes = link.read(serial_shared.bufferSize)
            if len(newBytes) == 0:
                continue

            remBytes += newBytes
            packets, remBytes = serial_shared.decodePackets(remBytes)
            for packetValues in packets:
                if firstTUsRaw is None:
                    firstTUsRaw = packetValues[0]
                rows.append(serial_shared.packetRow(packetValues, firstTUsRaw))

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
                    classifier.sampleRate,
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

            rawProbVector = classifier.predictRows(rows[-winRows:])
            elapsedSecs = now - startTime
            appendProbHistory(
                probTimes,
                probRows,
                elapsedSecs,
                rawProbVector,
            )

            probVector = smoothProbVector(
                probTimes,
                probRows,
                elapsedSecs,
            )
            state, confidence = ml.stateFromProb(probVector, labelNames)
            renderState(state, confidence, probVector, labelNames)
            updateProbFig(liveFigs["prob"], probTimes, probRows, labelNames)
            plotDirty = True

            if plotDirty and now - lastPlotEvent >= plotEventSecs:
                servicePlot(liveFigs["fig"])
                plotDirty = False
                lastPlotEvent = now


if __name__ == "__main__":
    main()
