################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import math
import struct
import sys
import time

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import serial

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
# Live mode loads the trained model and reads the same binary serial stream as
# capture.py. Change port here before running on the lab machine.
buildDir = Path(__file__).resolve().parents[1]
compactModelPath = buildDir / "outputs" / "models" / "motorPumpClassifier.joblib"
rawModelPath = buildDir / "outputs" / "models" / "motorPumpRawAxisClassifier.joblib"
port = "/dev/cu.usbserial-0001"
baudRate = 1000000
timeout = 1.0
expectedLabelNames = [
    "good",
    "bad_leak",
]

# Default live mode uses the compact DSP model.
# Run `python3 Build_v2/src/live.py raw` to load the raw-axis model.
modelMode = "compact"
if len(sys.argv) > 1:
    modelMode = sys.argv[1]
if modelMode not in ["compact", "raw"]:
    raise SystemExit("mode must be compact or raw")
if modelMode == "raw":
    modelPath = rawModelPath
else:
    modelPath = compactModelPath

# Firmware packet layout. This must match the struct sent by the MCU.
recordFormat = "<I12hf"
bufferSampleCount = 32
accelScale = 16384.0
gyroScale = 131.0
readyPrefix = "Sample struct size (bytes):"
startCommand = b"START\n"
recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount

# Plot and inference timing.
# The display can refresh faster than inference because plotting a signal is
# cheaper than building features and running the model.
plotSecs = 2.0
visualRefreshSecs = 0.1
plotEventSecs = 0.25
inferRefreshSecs = 0.5
probSmoothSecs = 5.0
probHistorySecs = 60.0
plotMaxHz = 500.0
magSensor = 1
fftSensor = 1

# Basic plot colours. These are only display settings and do not affect ML.
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


# Waits for the MCU ready message, then sends START to begin streaming.
def waitForReady(link):
    while True:
        line = link.readline().decode("ascii", errors="ignore").strip()
        if len(line) == 0:
            continue
        if line.startswith(readyPrefix):
            link.write(startCommand)
            link.flush()
            return


# Gives the serial connection a clean start and clears stale bytes.
def prepareLink(link):
    link.dtr = False
    link.rts = False
    time.sleep(0.2)
    link.reset_input_buffer()
    link.dtr = True
    link.rts = True


# Converts one firmware packet into the same row format used by captured CSVs.
# This keeps live inference and offline training using the same data layout.
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


# Applies the same dark plot style to each live axis.
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


# Turns the newest one-second live row buffer into one model feature row.
# This is the live version of the same path used in train.py.
def featureRowFromRows(rows, bundle):
    sampleRate = bundle["sampleRate"]
    df = pd.DataFrame(rows, columns=data.signalCols)
    df = data.cleanFrame(df)
    if bundle.get("modelMode") == "raw":
        return signals.rawModelInput(df)

    timeData = signals.timeSignals(df, sampleRate)
    freqData = signals.fftSignals(timeData, sampleRate, signals.fftConfig)
    return signals.modelInput(timeData, freqData)


# Finds the highest probability class from the model output.
def stateFromProb(probVector, labelNames):
    topIndex = int(np.argmax(probVector))
    state = labelNames[topIndex]
    confidence = float(probVector[topIndex])
    return state, confidence


# Prints the current smoothed state to the terminal.
# The escape sequence clears the terminal so the display updates in place.
def renderState(
    state,
    confidence,
    probVector,
    labelNames,
    rawState,
    rawConfidence,
    rawProbVector,
):
    lines = [
        f"state: {state}",
        f"confidence: {confidence * 100.0:.1f}%",
        f"raw state: {rawState}",
        f"raw confidence: {rawConfidence * 100.0:.1f}%",
        "",
        "raw probabilities:",
    ]
    for index, labelName in enumerate(labelNames):
        lines.append(f"{labelName}: {rawProbVector[index] * 100.0:.1f}%")

    lines += [
        "",
        "smoothed probabilities:",
    ]
    for index, labelName in enumerate(labelNames):
        lines.append(f"{labelName}: {probVector[index] * 100.0:.1f}%")
    text = "\n".join(lines)
    print(f"\x1b[2J\x1b[H{text}", end="", flush=True)


# Stops live inference if an old model with old labels is loaded by mistake.
def checkModelLabels(bundle):
    labelNames = list(bundle["labelNames"])
    if labelNames != expectedLabelNames:
        raise SystemExit(
            f"model labels are {labelNames}, expected {expectedLabelNames}"
        )
    if bundle.get("modelMode", "compact") != modelMode:
        raise SystemExit(
            f"model mode is {bundle.get('modelMode')}, expected {modelMode}"
        )

    modelValue = bundle["model"]
    classIndexes = list(modelValue.named_steps["logisticregression"].classes_)
    expectedIndexes = list(range(len(expectedLabelNames)))
    if classIndexes != expectedIndexes:
        raise SystemExit(
            f"model class indexes are {classIndexes}, expected {expectedIndexes}"
        )


# Builds the acceleration magnitude plot and returns the objects that need to be
# updated later.
def buildMagnitudeFig(ax):
    ax.set_title(f"MPU{magSensor} Acc Magnitude", color=textColor)
    ax.set_xlabel("time s", color=textColor)
    ax.set_ylabel("magnitude", color=textColor)
    line, = ax.plot([], [], color=magColor, linewidth=1.2)
    return {
        "ax": ax,
        "line": line,
    }


# Builds the FFT plot, including coloured BPFO/BPFI bands and a fundamental line.
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


# Builds the live probability history plot.
# Each label gets one line so changes over time can be seen during testing.
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


# Creates the three live figures: time magnitude, FFT, and model probability.
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


# Updates the time-domain magnitude plot using the newest displayed rows.
def updateMagnitudeFig(figData, df):
    t = df["t_s"].to_numpy(dtype=np.float64)
    y = signals.accelMagnitude(df, magSensor)
    t = t - t[0]
    figData["line"].set_data(t, y)
    figData["ax"].set_xlim(0.0, max(plotSecs, float(t[-1])))
    yMin = float(np.min(y))
    yMax = float(np.max(y))
    yPad = max(0.05, (yMax - yMin) * 0.1)
    figData["ax"].set_ylim(yMin - yPad, yMax + yPad)


# Updates the FFT plot and recalculates the displayed fundamental/BPFO/BPFI bands.
def updateFftFig(figData, df, sampleRate):
    hz, total = signals.accelFftSum(df, fftSensor, sampleRate)
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


# Updates the probability history plot from the stored raw probability rows.
def updateProbFig(figData, probTimes, probRows, labelNames):
    x = np.array(probTimes, dtype=np.float64)
    y = np.vstack(probRows).astype(np.float64) * 100.0
    for index in range(len(labelNames)):
        figData["lines"][index].set_data(x, y[:, index])
    if len(x) > 1:
        figData["ax"].set_xlim(max(0.0, x[-1] - probHistorySecs), x[-1])
    else:
        figData["ax"].set_xlim(0.0, probHistorySecs)


# Updates the signal plots from the latest rolling row buffer.
# It only uses the newest plotSecs of data so the graphs stay readable.
def updateVisualFigs(figs, rows, sampleRate, visualRows):
    viewRows = rows[-visualRows:]
    df = pd.DataFrame(viewRows, columns=data.signalCols)
    updateMagnitudeFig(figs["mag"], df)
    updateFftFig(figs["fft"], df, sampleRate)


# Lets matplotlib process redraw events without blocking the serial loop.
def servicePlot(fig):
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


# Stores probability history and drops old points outside the chart window.
def appendProbHistory(probTimes, probRows, elapsedSecs, probVector):
    probTimes.append(elapsedSecs)
    probRows.append(probVector)
    while len(probTimes) > 0 and probTimes[0] < elapsedSecs - probHistorySecs:
        del probTimes[0]
        del probRows[0]


# Averages recent probability vectors to stop the displayed state flickering.
# Raw probabilities are still plotted, but the text state uses this smoothed value.
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
    # Load the model bundle saved by train.py.
    # The bundle also provides the label order and one-second window size.
    bundle = joblib.load(modelPath)
    checkModelLabels(bundle)
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

    print("modelPath:", modelPath)
    print(
        "modelModified:",
        time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(modelPath.stat().st_mtime),
        ),
    )
    print("labels:", ", ".join(labelNames))
    print("modelMode:", bundle.get("modelMode", "compact"))
    print("sampleRate:", bundle["sampleRate"])
    print("winSecs:", bundle["winSecs"])
    print("probSmoothSecs:", probSmoothSecs)

    # Open the serial link and wait until the MCU says it is ready.
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
            # Read the next block of binary bytes from the MCU.
            # Empty reads are ignored so the loop can keep waiting.
            newBytes = link.read(bufferSize)
            if len(newBytes) == 0:
                continue

            remBytes += newBytes
            while len(remBytes) >= recordSize:
                # Decode all complete packets currently in the byte buffer.
                # Any incomplete packet stays in remBytes until more bytes arrive.
                packetBytes = remBytes[:recordSize]
                remBytes = remBytes[recordSize:]
                packetValues = struct.unpack(recordFormat, packetBytes)
                if firstTUsRaw is None:
                    firstTUsRaw = packetValues[0]
                rows.append(packetRow(packetValues, firstTUsRaw))

            # Keep only enough rows for the ML window and the visible plots.
            if len(rows) > keepRows:
                rows = rows[-keepRows:]
            if len(rows) < min(winRows, visualRows):
                continue

            now = time.perf_counter()
            if now - lastVisualUpdate >= visualRefreshSecs:
                # Signal plots can update as soon as enough rows are available.
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

            # Build one feature row from the latest one-second window and run
            # the model probability prediction.
            featureRow = featureRowFromRows(rows[-winRows:], bundle)
            rawProbVector = modelValue.predict_proba([featureRow])[0]
            rawState, rawConfidence = stateFromProb(rawProbVector, labelNames)
            elapsedSecs = now - startTime
            appendProbHistory(
                rawProbTimes,
                rawProbRows,
                elapsedSecs,
                rawProbVector,
            )

            # Smooth only the displayed text state. The probability chart still
            # shows raw model outputs over time.
            probVector = smoothProbVector(
                rawProbTimes,
                rawProbRows,
                elapsedSecs,
            )
            state, confidence = stateFromProb(probVector, labelNames)
            renderState(
                state,
                confidence,
                probVector,
                labelNames,
                rawState,
                rawConfidence,
                rawProbVector,
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
