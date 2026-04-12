################################################################################
# Imports                                                                      #
################################################################################
import matplotlib.pyplot as plt
import numpy as np

################################################################################
# variables/constants                                                          #
################################################################################
bgColor = [0.08, 0.03, 0.03]
textColor = [0.98, 0.95, 0.82]
closeColor = [0.98, 0.98, 0.98]
gridColor = [0.45, 0.40, 0.35]
accColor = closeColor
gyrColor = "gold"
tempColor = "deepskyblue"
fftColor = closeColor
bpfoColor = "tomato"
bpfiColor = "seagreen"
axisFftColor = closeColor

################################################################################
# helpers                                                                      #
################################################################################


# Applies the shared axis styling.
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


# Plots one time-domain signal on one axis.
def plotSigAx(ax, timeAxis, signal, color, title, ylabel):
    ax.plot(
        timeAxis,
        signal,
        color=color,
        linewidth=1.2,
    )
    ax.set_title(title, color=textColor, pad=8)
    ax.set_xlabel("time s", color=textColor)
    ax.set_ylabel(ylabel, color=textColor)


# Plots one acceleration FFT with the bearing bands and fundamental.
def plotAccFftAx(
    ax,
    freqAxis,
    spectrum,
    fundHz,
    fundMag,
    bpfoBand,
    bpfiBand,
    plotMaxHz,
    sensorLabel,
):
    plotMask = freqAxis <= plotMaxHz
    plotFreq = freqAxis[plotMask]
    plotMag = spectrum[plotMask]
    fullPeakIndex = int(np.argmax(plotMag))
    fullPeakHz = float(plotFreq[fullPeakIndex])
    fullPeakMag = float(plotMag[fullPeakIndex])
    ax.plot(
        plotFreq,
        plotMag,
        color=fftColor,
        linewidth=1.0,
    )
    ax.axvspan(bpfoBand[0], bpfoBand[1], color=bpfoColor, alpha=0.12)
    ax.axvspan(bpfiBand[0], bpfiBand[1], color=bpfiColor, alpha=0.10)
    ax.scatter([fundHz], [fundMag], color=bpfoColor, s=28, zorder=3)
    ax.scatter(
        [fullPeakHz],
        [fullPeakMag],
        color=tempColor,
        s=18,
        zorder=3,
    )
    ax.set_title(
        f"{sensorLabel} acc fft run {fundHz:.2f} hz full {fullPeakHz:.2f} hz",
        color=textColor,
        pad=8,
    )
    ax.set_xlabel("frequency hz", color=textColor)
    ax.set_ylabel("magnitude", color=textColor)
    ax.set_xlim(0.0, plotMaxHz)


# Plots one two-bar bearing RMS summary.
def plotBandAx(ax, bpfoRms, bpfiRms, sensorLabel):
    ax.bar(
        ["BPFO RMS", "BPFI RMS"],
        [bpfoRms, bpfiRms],
        width=0.8,
        color=[bpfoColor, bpfiColor],
        alpha=0.85,
    )
    ax.set_title(
        f"{sensorLabel} bearing band rms",
        color=textColor,
        pad=8,
    )
    ax.set_ylabel("rms magnitude", color=textColor)


# Plots one axis FFT spectrum with a marked fundamental.
def plotAxisAx(
    ax,
    freqAxis,
    spectrum,
    fundHz,
    fundMag,
    plotMaxHz,
    title,
    showXLabel,
    showYLabel,
):
    plotMask = freqAxis <= plotMaxHz
    plotFreq = freqAxis[plotMask]
    plotMag = spectrum[plotMask]
    fullPeakIndex = int(np.argmax(plotMag))
    fullPeakHz = float(plotFreq[fullPeakIndex])
    fullPeakMag = float(plotMag[fullPeakIndex])
    ax.plot(
        plotFreq,
        plotMag,
        color=axisFftColor,
        linewidth=1.0,
    )
    ax.scatter([fundHz], [fundMag], color=bpfoColor, s=18, zorder=3)
    ax.scatter(
        [fullPeakHz],
        [fullPeakMag],
        color=tempColor,
        s=14,
        zorder=3,
    )
    ax.set_title(
        f"{title} run {fundHz:.1f} full {fullPeakHz:.1f}",
        color=textColor,
        pad=6,
        fontsize=9,
    )
    ax.set_xlim(0.0, plotMaxHz)
    if showXLabel:
        ax.set_xlabel("frequency hz", color=textColor)
    if showYLabel:
        ax.set_ylabel("magnitude", color=textColor)


################################################################################
# main functions                                                               #
################################################################################


# Saves one sensor exploration figure for the current MPU block.
def plotSensor(rawSignals, timeSignals, freqSignals, savePath):
    plotMaxHz = 500.0
    sensorLabel = "MPU6050"

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(bgColor)
    grid = fig.add_gridspec(2, 3)
    accAx = fig.add_subplot(grid[0, 0])
    gyrAx = fig.add_subplot(grid[1, 0])
    fftAx = fig.add_subplot(grid[0, 1:])
    bandAx = fig.add_subplot(grid[1, 1])
    emptyAx = fig.add_subplot(grid[1, 2])

    styleAx(accAx)
    styleAx(gyrAx)
    styleAx(fftAx)
    styleAx(bandAx)
    fig.delaxes(emptyAx)

    plotSigAx(
        accAx,
        timeSignals["t_s"],
        timeSignals["accMag"],
        accColor,
        f"{sensorLabel} accMag",
        "magnitude",
    )
    plotSigAx(
        gyrAx,
        timeSignals["t_s"],
        timeSignals["gyrMag"],
        gyrColor,
        f"{sensorLabel} gyroMag",
        "magnitude",
    )
    plotAccFftAx(
        fftAx,
        freqSignals["freqAxis"],
        freqSignals["accSpectrum"],
        freqSignals["accFundHz"],
        freqSignals["accFundMag"],
        freqSignals["accBpfoBand"],
        freqSignals["accBpfiBand"],
        plotMaxHz,
        sensorLabel,
    )
    plotBandAx(
        bandAx,
        freqSignals["accBpfoRms"],
        freqSignals["accBpfiRms"],
        sensorLabel,
    )

    fig.tight_layout(pad=0.8)
    plt.savefig(savePath, facecolor=bgColor)
    plt.close(fig)


# Saves one 6-panel FFT exploration figure for all accel and gyro axes.
def plotAxisGrid(freqSignals, savePath):
    plotMaxHz = 500.0
    signalGrid = [
        ("accX", "Acc X FFT"),
        ("accY", "Acc Y FFT"),
        ("accZ", "Acc Z FFT"),
        ("gyrX", "Gyr X FFT"),
        ("gyrY", "Gyr Y FFT"),
        ("gyrZ", "Gyr Z FFT"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor(bgColor)

    for plotIndex, (signalName, title) in enumerate(signalGrid):
        rowIndex = plotIndex // 3
        colIndex = plotIndex % 3
        ax = axes[rowIndex, colIndex]
        styleAx(ax)
        plotAxisAx(
            ax,
            freqSignals["freqAxis"],
            freqSignals[f"{signalName}Spectrum"],
            freqSignals[f"{signalName}FundHz"],
            freqSignals[f"{signalName}FundMag"],
            plotMaxHz,
            title,
            rowIndex == 1,
            colIndex == 0,
        )

    fig.tight_layout(pad=1.0)
    plt.savefig(savePath, facecolor=bgColor)
    plt.close(fig)
