import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG_COLOR = [0.08, 0.03, 0.03]
TEXT_COLOR = [0.98, 0.95, 0.82]
CLOSE_COLOR = [0.98, 0.98, 0.98]
GRID_COLOR = [0.45, 0.40, 0.35]
DB_FLOOR = 1e-12
HARMONIC_COUNT = 8
FUNDAMENTAL_WINDOW_DB = 20.0
FFT_PLOT_MAX_HZ = 50.0


# Applies the shared axis styling.
def _styleAx(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(
        color=GRID_COLOR,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )


# Applies the shared legend styling.
def _styleLeg(legend):
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    frame = legend.get_frame()
    frame.set_facecolor(BG_COLOR)
    frame.set_edgecolor(GRID_COLOR)


# Converts one magnitude signal into dB.
def magDb(signal):
    magnitudeDb = 20.0 * np.log10(np.maximum(signal, DB_FLOOR))
    return magnitudeDb


# Converts one magnitude value into dB.
def magValDb(magnitudeValue):
    magnitudeValueDb = 20.0 * np.log10(max(magnitudeValue, DB_FLOOR))
    return magnitudeValueDb


# Converts one energy value into dB.
def engDb(energyValue):
    energyDb = 10.0 * np.log10(max(energyValue, DB_FLOOR))
    return energyDb


# Builds the plotted band labels from the configured frequency bands.
def bandTicks(frequencyBands):
    bandTickLabels = []
    for _, bandLow, bandHigh in frequencyBands:
        bandTickLabels.append(f"{bandLow}-{bandHigh}")
    return bandTickLabels


# Builds the fundamental index from one FFT magnitude spectrum.
def fundIdx(spectrum):
    spectrumDb = magDb(spectrum)
    peakDb = np.max(spectrumDb[1:])
    thresholdDb = peakDb - FUNDAMENTAL_WINDOW_DB
    candidateIndexes = np.where(spectrumDb >= thresholdDb)[0]
    fundamentalIndex = candidateIndexes[candidateIndexes > 0][0]
    return fundamentalIndex


# Builds one harmonic frequency and magnitude set from one spectrum.
def harmSeries(frequencyAxis, spectrum, harmonicCount):
    fundamentalIndex = fundIdx(spectrum)
    fundamentalFrequency = frequencyAxis[fundamentalIndex]
    harmonicFrequencies = []
    harmonicMagnitudesDb = []
    for harmonicNumber in range(1, harmonicCount + 1):
        targetFrequency = fundamentalFrequency * harmonicNumber
        if targetFrequency > frequencyAxis[-1]:
            break
        harmonicIndex = int(np.argmin(np.abs(frequencyAxis - targetFrequency)))
        harmonicFrequencies.append(frequencyAxis[harmonicIndex])
        harmonicMagnitudesDb.append(
            magValDb(spectrum[harmonicIndex])
        )
    return np.array(harmonicFrequencies), np.array(harmonicMagnitudesDb)


# Plots a list of available time-domain lines on one axis.
def plotSigGrp(ax, timeSignals, signalGroup, title, ylabel):
    for signalKey, color in signalGroup:
        samplePlot = timeSignals["sample"][:100]
        signalPlot = timeSignals[signalKey][:100]
        ax.plot(
            samplePlot,
            signalPlot,
            color=color,
            linewidth=1.2,
            label=signalKey,
        )
    ax.set_title(title, color=TEXT_COLOR, pad=8)
    ax.set_xlabel("sample", color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    legend = ax.legend(loc="best", frameon=True, fontsize=8)
    _styleLeg(legend)


# Plots one FFT harmonic series group on one axis.
def plotFftGrp(ax, frequencyAxis, freqSignals, signalGroup, title):
    harmonicSeries = []
    for signalKey, color in signalGroup:
        harmonicFrequencies, harmonicMagnitudesDb = harmSeries(
            frequencyAxis,
            freqSignals[signalKey],
            HARMONIC_COUNT,
        )
        harmonicSeries.append(
            (
                signalKey,
                color,
                harmonicFrequencies,
                harmonicMagnitudesDb,
            )
        )

    minDb = min(np.min(series[3]) for series in harmonicSeries) - 6.0
    maxDb = max(np.max(series[3]) for series in harmonicSeries) + 3.0
    for signalKey, color, harmonicFrequencies, harmonicMagnitudesDb in (
        harmonicSeries
    ):
        ax.vlines(
            harmonicFrequencies,
            minDb,
            harmonicMagnitudesDb,
            color=color,
            alpha=0.35,
            linewidth=1.0,
        )
        ax.plot(
            harmonicFrequencies,
            harmonicMagnitudesDb,
            color=color,
            linewidth=1.2,
            marker="o",
            label=signalKey,
        )
        ax.axvline(
            harmonicFrequencies[0],
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
    ax.set_title(title, color=TEXT_COLOR)
    ax.set_xlabel("frequency hz", color=TEXT_COLOR)
    ax.set_ylabel("magnitude db", color=TEXT_COLOR)
    ax.set_xlim(0, min(FFT_PLOT_MAX_HZ, frequencyAxis[-1]))
    ax.set_ylim(minDb, maxDb)
    legend = ax.legend(loc="best", frameon=True, fontsize=8)
    _styleLeg(legend)


# Builds one grouped set of band energies for the available signals.
def bandGrps(freqSignals, frequencyBands):
    bandGroups = []
    for signalPrefix, color in [
        ("mpu1Acc", CLOSE_COLOR),
        ("mpu2Acc", "deepskyblue"),
        ("mpu1Gyr", "gold"),
        ("mpu2Gyr", "tomato"),
    ]:
        bandValues = []
        for bandName, _, _ in frequencyBands:
            bandKey = f"{signalPrefix}AxisPower{bandName}"
            bandValues.append(engDb(freqSignals[bandKey]))
        bandGroups.append((signalPrefix, color, bandValues))
    return bandGroups


# Plots one grouped band-energy chart for the available signals.
def plotBandGrps(ax, frequencyBands, bandGroups, title):
    groupCount = len(bandGroups)
    bandTickPositions = []
    bandTickLabels = bandTicks(frequencyBands)
    for bandIndex, (_, bandLow, bandHigh) in enumerate(frequencyBands):
        bandWidth = bandHigh - bandLow
        bandTickPositions.append(bandLow + (bandWidth / 2.0))
        if groupCount == 0:
            continue
        groupedBarWidth = (bandWidth / groupCount) * 0.9
        for groupIndex, (
            signalName,
            color,
            bandValues,
        ) in enumerate(bandGroups):
            barLeft = bandLow + ((bandWidth / groupCount) * groupIndex)
            ax.bar(
                barLeft,
                bandValues[bandIndex],
                width=groupedBarWidth,
                align="edge",
                color=color,
                label=signalName if bandIndex == 0 else None,
            )
    ax.set_title(title, color=TEXT_COLOR)
    ax.set_xlabel("frequency hz", color=TEXT_COLOR)
    ax.set_ylabel("energy db", color=TEXT_COLOR)
    ax.set_xticks(bandTickPositions, bandTickLabels)
    ax.set_xlim(frequencyBands[0][1], frequencyBands[-1][2])
    if groupCount > 0:
        legend = ax.legend(loc="best", frameon=True, fontsize=8)
        _styleLeg(legend)


# Plots the current time-domain magnitude and temperature-average signals.
def plotRaw(signalData, savePath):
    timeSignals = signalData["timeSignals"]

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(2, 2)
    accAx = fig.add_subplot(grid[0, 0])
    gyrAx = fig.add_subplot(grid[0, 1])
    tempAx = fig.add_subplot(grid[1, :])

    _styleAx(accAx)
    _styleAx(gyrAx)
    _styleAx(tempAx)

    plotSigGrp(
        accAx,
        timeSignals,
        [
            ("mpu1AccMag", CLOSE_COLOR),
            ("mpu2AccMag", "deepskyblue"),
        ],
        "accel magnitude vs sample",
        "accel mag",
    )
    plotSigGrp(
        gyrAx,
        timeSignals,
        [
            ("mpu1GyrMag", CLOSE_COLOR),
            ("mpu2GyrMag", "deepskyblue"),
        ],
        "gyro magnitude vs sample",
        "gyro mag",
    )
    plotSigGrp(
        tempAx,
        timeSignals,
        [
            ("mpu1TempAvg", CLOSE_COLOR),
            ("mpu2TempAvg", "deepskyblue"),
            ("ds18b20OneAvg", "gold"),
            ("ds18b20TwoAvg", "tomato"),
        ],
        "temperature average vs sample",
        "temperature",
    )

    fig.tight_layout(pad=0.8)
    plt.savefig(savePath, facecolor=BG_COLOR)
    plt.close(fig)


# Plots FFT harmonics and grouped band-energy bars.
def plotFrequency(signalData, savePath):
    freqSignals = signalData["freqSignals"]
    frequencyAxis = freqSignals["frequencyAxis"]
    frequencyBands = signalData["signalMeta"]["frequencyBands"]
    bandGroups = bandGrps(freqSignals, frequencyBands)

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(2, 2)
    accSpectrumAx = fig.add_subplot(grid[0, 0])
    gyrSpectrumAx = fig.add_subplot(grid[0, 1])
    accBandAx = fig.add_subplot(grid[1, 0])
    gyrBandAx = fig.add_subplot(grid[1, 1])

    _styleAx(accSpectrumAx)
    _styleAx(gyrSpectrumAx)
    _styleAx(accBandAx)
    _styleAx(gyrBandAx)

    plotFftGrp(
        accSpectrumAx,
        frequencyAxis,
        freqSignals,
        [
            ("mpu1AccMagSpectrum", CLOSE_COLOR),
            ("mpu2AccMagSpectrum", "deepskyblue"),
        ],
        "accel fft fundamentals and harmonics",
    )

    plotFftGrp(
        gyrSpectrumAx,
        frequencyAxis,
        freqSignals,
        [
            ("mpu1GyrMagSpectrum", CLOSE_COLOR),
            ("mpu2GyrMagSpectrum", "deepskyblue"),
        ],
        "gyro fft fundamentals and harmonics",
    )

    plotBandGrps(
        accBandAx,
        frequencyBands,
        [
            bandGroup
            for bandGroup in bandGroups
            if bandGroup[0] in ["mpu1Acc", "mpu2Acc"]
        ],
        "accel band energy",
    )
    plotBandGrps(
        gyrBandAx,
        frequencyBands,
        [
            bandGroup
            for bandGroup in bandGroups
            if bandGroup[0] in ["mpu1Gyr", "mpu2Gyr"]
        ],
        "gyro band energy",
    )

    fig.tight_layout(pad=0.8)
    plt.savefig(savePath, facecolor=BG_COLOR)
    plt.close(fig)
