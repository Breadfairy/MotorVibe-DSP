from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft


# Builds one DC-removed amplitude spectrum for one signal.
def buildSpectrum(signal, sampleRate):
    sampleCount = signal.size
    centeredSignal = signal - np.mean(signal)
    spectrum = fft.rfft(centeredSignal)
    magnitude = (2.0 / sampleCount) * np.abs(spectrum)
    frequencyAxis = fft.rfftfreq(sampleCount, d=1.0 / sampleRate)
    return frequencyAxis, magnitude


# Finds one dominant peak inside the selected frequency window.
def fundamentalPeak(frequencyAxis, magnitude, minFundHz, maxFundHz):
    fundMask = (
        (frequencyAxis >= minFundHz)
        & (frequencyAxis <= maxFundHz)
    )
    fundFrequencies = frequencyAxis[fundMask]
    fundMagnitudes = magnitude[fundMask]
    peakIndex = int(np.argmax(fundMagnitudes))
    peakFrequency = float(fundFrequencies[peakIndex])
    peakMagnitude = float(fundMagnitudes[peakIndex])
    return peakFrequency, peakMagnitude


# Plots the first time window for one channel.
def plotTimeAx(ax, timeAxis, signal, axisName, timePlotSeconds):
    timeMask = timeAxis <= timePlotSeconds
    ax.plot(
        timeAxis[timeMask],
        signal[timeMask],
        color="deepskyblue",
        linewidth=1.0,
    )
    ax.set_title(axisName)
    ax.set_xlabel("time s")
    ax.set_ylabel("value")
    ax.grid(
        linestyle=":",
        linewidth=0.6,
        alpha=0.8,
    )


# Plots one FFT magnitude line with one marked fundamental.
def plotFftAx(
    ax,
    frequencyAxis,
    magnitude,
    fundamentalFrequency,
    fundamentalMagnitude,
    bpfoLowHz,
    bpfoHighHz,
    bpfiLowHz,
    bpfiHighHz,
    minPlotHz,
    maxPlotHz,
):
    plotMask = (
        (frequencyAxis >= minPlotHz)
        & (frequencyAxis <= maxPlotHz)
    )
    plotFrequencies = frequencyAxis[plotMask]
    plotMagnitudes = magnitude[plotMask]
    plotPeak = float(np.max(plotMagnitudes))
    ax.plot(
        plotFrequencies,
        plotMagnitudes,
        color="deepskyblue",
        linewidth=1.0,
    )
    ax.axvspan(
        bpfoLowHz,
        bpfoHighHz,
        color="tomato",
        alpha=0.12,
    )
    ax.axvspan(
        bpfiLowHz,
        bpfiHighHz,
        color="seagreen",
        alpha=0.10,
    )
    ax.scatter(
        [fundamentalFrequency],
        [plotPeak],
        color="tomato",
        s=28,
        zorder=3,
    )
    ax.set_title(f"Fundamental Hz ({fundamentalFrequency:.2f} Hz)")
    ax.set_xlabel("frequency hz")
    ax.set_ylabel("magnitude")
    ax.set_xlim(minPlotHz, maxPlotHz)
    ax.grid(
        linestyle=":",
        linewidth=0.6,
        alpha=0.8,
    )


# Builds one RMS magnitude for one target band.
def bandRms(frequencyAxis, magnitude, bandLowHz, bandHighHz):
    bandMask = (
        (frequencyAxis >= bandLowHz)
        & (frequencyAxis <= bandHighHz)
    )
    bandMagnitudes = magnitude[bandMask]
    bandRmsValue = float(np.sqrt(np.mean(bandMagnitudes ** 2)))
    return bandRmsValue


# Builds one order band with a symmetric tolerance.
def orderBand(fundamentalFrequency, lowOrder, highOrder, toleranceFraction):
    bandLowHz = (fundamentalFrequency * lowOrder) * (
        1.0 - toleranceFraction
    )
    bandHighHz = (fundamentalFrequency * highOrder) * (
        1.0 + toleranceFraction
    )
    return bandLowHz, bandHighHz


# Plots one two-bar RMS summary for BPFO and BPFI.
def plotBandRmsAx(ax, axisName, bpfoBandRms, bpfiBandRms):
    ax.bar(
        ["BPFO RMS", "BPFI RMS"],
        [bpfoBandRms, bpfiBandRms],
        width=0.8,
        color=["tomato", "seagreen"],
        alpha=0.85,
    )
    ax.set_title("Bearing Band RMS")
    ax.set_ylabel("rms magnitude")
    if axisName == "Axis1":
        ax.set_ylim(0.0, 0.001)
    elif axisName == "Axis2":
        ax.set_ylim(0.0, 0.01)
    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.8,
    )


# Builds the display axis name from one channel key.
def axisName(channelKey):
    axisNames = {
        "channel0": "Axis1",
        "channel1": "Axis2",
    }
    return axisNames.get(channelKey, channelKey)


# Runs one FFT explorer for exported .npy capture CSV files.
def runHarness():
    repoRoot = Path(__file__).resolve().parents[1]
    dataDir = repoRoot / "data" / "testdata"
    sampleRate = 32000.0
    timePlotSeconds = 1.0
    minFundHz = 1.0
    minPlotHz = 1.0
    maxPlotHz = 130.0
    bpfoLowOrder = 3.0
    bpfoHighOrder = 5.0
    bpfiLowOrder = 5.0
    bpfiHighOrder = 7.0
    orderToleranceFraction = 0.10
    csvPaths = sorted(dataDir.glob("data_v2-10s-record*.csv"))

    print("sampleRateHz:", sampleRate)
    print("timePlotSeconds:", timePlotSeconds)
    print("minFundHz:", minFundHz)
    print("maxPlotHz:", maxPlotHz)

    for csvPath in csvPaths:
        dataFrame = pd.read_csv(csvPath)
        channelKeys = [
            columnName
            for columnName in dataFrame.columns
            if columnName not in ["sample", "timeSeconds"]
        ]
        savePath = dataDir / f"{csvPath.stem}_fft.png"
        sampleCount = dataFrame.shape[0]
        durationSeconds = sampleCount / sampleRate
        frequencyResolutionHz = sampleRate / sampleCount
        timeAxis = dataFrame["timeSeconds"].to_numpy(dtype=np.float64)

        fig, axes = plt.subplots(
            len(channelKeys),
            3,
            figsize=(16, 4 * len(channelKeys)),
            squeeze=False,
        )

        for channelIndex, channelKey in enumerate(channelKeys):
            currentAxisName = axisName(channelKey)
            signal = dataFrame[channelKey].to_numpy(dtype=np.float64)
            frequencyAxis, magnitude = buildSpectrum(
                signal,
                sampleRate,
            )
            fundamentalFrequency, fundamentalMagnitude = fundamentalPeak(
                frequencyAxis,
                magnitude,
                minFundHz,
                3000.0,
            )
            shaftRpm = fundamentalFrequency * 60.0
            bpfoLowHz, bpfoHighHz = orderBand(
                fundamentalFrequency,
                bpfoLowOrder,
                bpfoHighOrder,
                orderToleranceFraction,
            )
            bpfoBandRms = bandRms(
                frequencyAxis,
                magnitude,
                bpfoLowHz,
                bpfoHighHz,
            )
            bpfiLowHz, bpfiHighHz = orderBand(
                fundamentalFrequency,
                bpfiLowOrder,
                bpfiHighOrder,
                orderToleranceFraction,
            )
            bpfiBandRms = bandRms(
                frequencyAxis,
                magnitude,
                bpfiLowHz,
                bpfiHighHz,
            )

            plotTimeAx(
                axes[channelIndex, 0],
                timeAxis,
                signal,
                currentAxisName,
                timePlotSeconds,
            )
            plotFftAx(
                axes[channelIndex, 1],
                frequencyAxis,
                magnitude,
                fundamentalFrequency,
                fundamentalMagnitude,
                bpfoLowHz,
                bpfoHighHz,
                bpfiLowHz,
                bpfiHighHz,
                minPlotHz,
                maxPlotHz,
            )
            plotBandRmsAx(
                axes[channelIndex, 2],
                currentAxisName,
                bpfoBandRms,
                bpfiBandRms,
            )

            print(
                f"{csvPath.name} {currentAxisName}: "
                f"fundamentalFrequencyHz={fundamentalFrequency:.4f} "
                f"fundamentalMagnitude={fundamentalMagnitude:.6f}"
            )
            print(
                f"{csvPath.name} {currentAxisName} shaftRpm={shaftRpm:.2f} "
                f"bpfoBandHz={bpfoLowHz:.2f}-{bpfoHighHz:.2f} "
                f"bpfoBandRms={bpfoBandRms:.6f}",
            )
            print(
                f"{csvPath.name} {currentAxisName} "
                f"bpfiBandHz={bpfiLowHz:.2f}-{bpfiHighHz:.2f} "
                f"bpfiBandRms={bpfiBandRms:.6f}",
            )

        fig.suptitle("TestData Frequency analysis")
        fig.tight_layout()
        plt.savefig(savePath)
        plt.close(fig)

        print("csvPath:", str(csvPath))
        print("sampleCount:", sampleCount)
        print("durationSeconds:", durationSeconds)
        print("frequencyResolutionHz:", frequencyResolutionHz)
        print("savePath:", str(savePath))


if __name__ == "__main__":
    runHarness()
