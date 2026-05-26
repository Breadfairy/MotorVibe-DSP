################################################################################
# Imports                                                                      #
################################################################################
import matplotlib.pyplot as plt

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
    ax.plot(
        freqAxis[plotMask],
        spectrum[plotMask],
        color=fftColor,
        linewidth=1.0,
    )
    ax.axvspan(bpfoBand[0], bpfoBand[1], color=bpfoColor, alpha=0.12)
    ax.axvspan(bpfiBand[0], bpfiBand[1], color=bpfiColor, alpha=0.10)
    ax.scatter([fundHz], [fundMag], color=bpfoColor, s=28, zorder=3)
    ax.set_title(
        f"{sensorLabel} acc fft ({fundHz:.2f} hz)",
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
    ax.plot(
        freqAxis[plotMask],
        spectrum[plotMask],
        color=axisFftColor,
        linewidth=1.0,
    )
    ax.scatter([fundHz], [fundMag], color=bpfoColor, s=18, zorder=3)
    ax.set_title(title, color=textColor, pad=6, fontsize=9)
    ax.set_xlim(0.0, plotMaxHz)
    if showXLabel:
        ax.set_xlabel("frequency hz", color=textColor)
    if showYLabel:
        ax.set_ylabel("magnitude", color=textColor)


################################################################################
# main functions                                                               #
################################################################################


# Saves one sensor exploration figure for one MPU block.
def plotSensor(rawSignals, timeSignals, freqSignals, sensorNumber, savePath):
    plotMaxHz = 500.0
    sensorLabel = f"MPU6050 {sensorNumber}"
    if sensorNumber == 1:
        accMagKey = "acc1Mag"
        gyrMagKey = "gyr1Mag"
        accSpectrumKey = "acc1Spectrum"
        fundHzKey = "acc1FundHz"
        fundMagKey = "acc1FundMag"
        bpfoBandKey = "acc1BpfoBand"
        bpfiBandKey = "acc1BpfiBand"
        bpfoRmsKey = "acc1BpfoRms"
        bpfiRmsKey = "acc1BpfiRms"
    else:
        accMagKey = "acc2Mag"
        gyrMagKey = "gyr2Mag"
        accSpectrumKey = "acc2Spectrum"
        fundHzKey = "acc2FundHz"
        fundMagKey = "acc2FundMag"
        bpfoBandKey = "acc2BpfoBand"
        bpfiBandKey = "acc2BpfiBand"
        bpfoRmsKey = "acc2BpfoRms"
        bpfiRmsKey = "acc2BpfiRms"

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(bgColor)
    grid = fig.add_gridspec(2, 3)
    accAx = fig.add_subplot(grid[0, 0])
    gyrAx = fig.add_subplot(grid[1, 0])
    tempAx = fig.add_subplot(grid[0, 1])
    fftAx = fig.add_subplot(grid[0, 2])
    bandAx = fig.add_subplot(grid[1, 1])
    emptyAx = fig.add_subplot(grid[1, 2])

    styleAx(accAx)
    styleAx(gyrAx)
    styleAx(tempAx)
    styleAx(fftAx)
    styleAx(bandAx)
    fig.delaxes(emptyAx)

    plotSigAx(
        accAx,
        timeSignals["t_s"],
        timeSignals[accMagKey],
        accColor,
        f"{sensorLabel} accMag",
        "magnitude",
    )
    plotSigAx(
        gyrAx,
        timeSignals["t_s"],
        timeSignals[gyrMagKey],
        gyrColor,
        f"{sensorLabel} gyroMag",
        "magnitude",
    )
    plotSigAx(
        tempAx,
        timeSignals["t_s"],
        rawSignals["tempC"],
        tempColor,
        "Temp C",
        "temperature",
    )
    plotAccFftAx(
        fftAx,
        freqSignals["freqAxis"],
        freqSignals[accSpectrumKey],
        freqSignals[fundHzKey],
        freqSignals[fundMagKey],
        freqSignals[bpfoBandKey],
        freqSignals[bpfiBandKey],
        plotMaxHz,
        sensorLabel,
    )
    plotBandAx(
        bandAx,
        freqSignals[bpfoRmsKey],
        freqSignals[bpfiRmsKey],
        sensorLabel,
    )

    fig.tight_layout(pad=0.8)
    plt.savefig(savePath, facecolor=bgColor)
    plt.close(fig)


# Saves one 12-panel FFT exploration figure for all accel and gyro axes.
def plotAxisGrid(freqSignals, savePath):
    plotMaxHz = 500.0
    signalGrid = [
        ("acc1X", "MPU1 Acc X FFT"),
        ("acc1Y", "MPU1 Acc Y FFT"),
        ("acc1Z", "MPU1 Acc Z FFT"),
        ("gyr1X", "MPU1 Gyr X FFT"),
        ("gyr1Y", "MPU1 Gyr Y FFT"),
        ("gyr1Z", "MPU1 Gyr Z FFT"),
        ("acc2X", "MPU2 Acc X FFT"),
        ("acc2Y", "MPU2 Acc Y FFT"),
        ("acc2Z", "MPU2 Acc Z FFT"),
        ("gyr2X", "MPU2 Gyr X FFT"),
        ("gyr2Y", "MPU2 Gyr Y FFT"),
        ("gyr2Z", "MPU2 Gyr Z FFT"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
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
            rowIndex == 3,
            colIndex == 0,
        )

    fig.tight_layout(pad=1.0)
    plt.savefig(savePath, facecolor=bgColor)
    plt.close(fig)
