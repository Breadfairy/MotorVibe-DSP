import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG_COLOR = [0.08, 0.03, 0.03]
TEXT_COLOR = [0.98, 0.95, 0.82]
CLOSE_COLOR = [0.98, 0.98, 0.98]
GRID_COLOR = [0.45, 0.40, 0.35]
ACC_COLOR = CLOSE_COLOR
GYR_COLOR = "gold"
MPU_TEMP_COLOR = "tomato"
DS_TEMP_COLOR = "deepskyblue"
FFT_COLOR = CLOSE_COLOR
BPFO_COLOR = "tomato"
BPFI_COLOR = "seagreen"
AXIS_FFT_COLOR = CLOSE_COLOR


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


# Builds the sample slice for the selected time plot window.
def timeSlice(sampleRate, sampleSize, plotSeconds):
    timePlotSampleCount = int(sampleRate * plotSeconds)
    plotStop = min(timePlotSampleCount, sampleSize)
    return slice(0, plotStop)


# Plots one time-domain signal on one axis.
def plotSigAx(ax, sample, signal, color, title, ylabel):
    ax.plot(
        sample,
        signal,
        color=color,
        linewidth=1.2,
    )
    ax.set_title(title, color=TEXT_COLOR, pad=8)
    ax.set_xlabel("sample", color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)


# Plots one acceleration FFT with the bearing bands and fundamental.
def plotAccFftAx(
    ax,
    frequencyAxis,
    spectrum,
    fundamentalHz,
    fundamentalMag,
    bpfoBand,
    bpfiBand,
    plotMaxHz,
    sensorLabel,
):
    plotMask = frequencyAxis <= plotMaxHz
    ax.plot(
        frequencyAxis[plotMask],
        spectrum[plotMask],
        color=FFT_COLOR,
        linewidth=1.0,
    )
    ax.axvspan(
        bpfoBand[0],
        bpfoBand[1],
        color=BPFO_COLOR,
        alpha=0.12,
    )
    ax.axvspan(
        bpfiBand[0],
        bpfiBand[1],
        color=BPFI_COLOR,
        alpha=0.10,
    )
    ax.scatter(
        [fundamentalHz],
        [fundamentalMag],
        color=BPFO_COLOR,
        s=28,
        zorder=3,
    )
    ax.set_title(
        f"{sensorLabel} acc fft ({fundamentalHz:.2f} hz)",
        color=TEXT_COLOR,
        pad=8,
    )
    ax.set_xlabel("frequency hz", color=TEXT_COLOR)
    ax.set_ylabel("magnitude", color=TEXT_COLOR)
    ax.set_xlim(0.0, plotMaxHz)


# Plots one two-bar bearing RMS summary.
def plotBandRmsAx(ax, bpfoBandRms, bpfiBandRms, sensorLabel):
    ax.bar(
        ["BPFO RMS", "BPFI RMS"],
        [bpfoBandRms, bpfiBandRms],
        width=0.8,
        color=[BPFO_COLOR, BPFI_COLOR],
        alpha=0.85,
    )
    ax.set_title(
        f"{sensorLabel} bearing band rms",
        color=TEXT_COLOR,
        pad=8,
    )
    ax.set_ylabel("rms magnitude", color=TEXT_COLOR)


# Plots one axis FFT spectrum with a marked fundamental.
def plotAxisFftAx(
    ax,
    frequencyAxis,
    spectrum,
    fundamentalHz,
    fundamentalMag,
    plotMaxHz,
    title,
    showXLabel,
    showYLabel,
):
    plotMask = frequencyAxis <= plotMaxHz
    ax.plot(
        frequencyAxis[plotMask],
        spectrum[plotMask],
        color=AXIS_FFT_COLOR,
        linewidth=1.0,
    )
    ax.scatter(
        [fundamentalHz],
        [fundamentalMag],
        color=BPFO_COLOR,
        s=18,
        zorder=3,
    )
    ax.set_title(title, color=TEXT_COLOR, pad=6, fontsize=9)
    ax.set_xlim(0.0, plotMaxHz)
    if showXLabel:
        ax.set_xlabel("frequency hz", color=TEXT_COLOR)
    else:
        ax.set_xlabel("")
    if showYLabel:
        ax.set_ylabel("magnitude", color=TEXT_COLOR)
    else:
        ax.set_ylabel("")


# Plots one sensor exploration figure for one MPU sensor block.
def plotSensor(signalData, sensorNumber, savePath, plotSeconds):
    sampleRate = signalData["signalMeta"]["sampleRate"]
    plotMaxHz = signalData["signalMeta"]["fftConfig"]["plotMaxHz"]
    rawSignals = signalData["rawSignals"]
    timeSignals = signalData["timeSignals"]
    freqSignals = signalData["freqSignals"]
    plotSlice = timeSlice(
        sampleRate,
        rawSignals["sample"].size,
        plotSeconds,
    )

    sensorLabel = f"MPU6050 {sensorNumber}"
    if sensorNumber == 1:
        accMagKey = "mpu1AccMag"
        gyrMagKey = "mpu1GyrMag"
        mpuTempKey = "mpu1Temp"
        dsTempKey = "ds18b20One"
        accSpectrumKey = "mpu1AccSpectrum"
        fundamentalHzKey = "mpu1AccFundamentalHz"
        fundamentalMagKey = "mpu1AccFundamentalMag"
        bpfoBandKey = "mpu1AccBpfoBand"
        bpfiBandKey = "mpu1AccBpfiBand"
        bpfoBandRmsKey = "mpu1AccBpfoBandRms"
        bpfiBandRmsKey = "mpu1AccBpfiBandRms"
    else:
        accMagKey = "mpu2AccMag"
        gyrMagKey = "mpu2GyrMag"
        mpuTempKey = "mpu2Temp"
        dsTempKey = "ds18b20Two"
        accSpectrumKey = "mpu2AccSpectrum"
        fundamentalHzKey = "mpu2AccFundamentalHz"
        fundamentalMagKey = "mpu2AccFundamentalMag"
        bpfoBandKey = "mpu2AccBpfoBand"
        bpfiBandKey = "mpu2AccBpfiBand"
        bpfoBandRmsKey = "mpu2AccBpfoBandRms"
        bpfiBandRmsKey = "mpu2AccBpfiBandRms"

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(2, 3)
    accAx = fig.add_subplot(grid[0, 0])
    gyrAx = fig.add_subplot(grid[1, 0])
    mpuTempAx = fig.add_subplot(grid[0, 1])
    dsTempAx = fig.add_subplot(grid[1, 1])
    fftAx = fig.add_subplot(grid[0, 2])
    bandAx = fig.add_subplot(grid[1, 2])

    _styleAx(accAx)
    _styleAx(gyrAx)
    _styleAx(mpuTempAx)
    _styleAx(dsTempAx)
    _styleAx(fftAx)
    _styleAx(bandAx)

    plotSigAx(
        accAx,
        rawSignals["sample"][plotSlice],
        timeSignals[accMagKey][plotSlice],
        ACC_COLOR,
        f"{sensorLabel} accMag",
        "magnitude",
    )
    plotSigAx(
        gyrAx,
        rawSignals["sample"][plotSlice],
        timeSignals[gyrMagKey][plotSlice],
        GYR_COLOR,
        f"{sensorLabel} gyroMag",
        "magnitude",
    )
    plotSigAx(
        mpuTempAx,
        rawSignals["sample"][plotSlice],
        rawSignals[mpuTempKey][plotSlice],
        MPU_TEMP_COLOR,
        f"{sensorLabel} mpu temp",
        "temperature",
    )
    plotSigAx(
        dsTempAx,
        rawSignals["sample"][plotSlice],
        rawSignals[dsTempKey][plotSlice],
        DS_TEMP_COLOR,
        f"{sensorLabel} ds18b20 temp",
        "temperature",
    )
    plotAccFftAx(
        fftAx,
        freqSignals["frequencyAxis"],
        freqSignals[accSpectrumKey],
        freqSignals[fundamentalHzKey],
        freqSignals[fundamentalMagKey],
        freqSignals[bpfoBandKey],
        freqSignals[bpfiBandKey],
        plotMaxHz,
        sensorLabel,
    )
    plotBandRmsAx(
        bandAx,
        freqSignals[bpfoBandRmsKey],
        freqSignals[bpfiBandRmsKey],
        sensorLabel,
    )

    fig.tight_layout(pad=0.8)
    plt.savefig(savePath, facecolor=BG_COLOR)
    plt.close(fig)


# Plots one 12-panel FFT exploration figure for all accel and gyro axes.
def plotAxisFftGrid(signalData, savePath):
    plotMaxHz = signalData["signalMeta"]["fftConfig"]["plotMaxHz"]
    freqSignals = signalData["freqSignals"]
    frequencyAxis = freqSignals["frequencyAxis"]
    signalGrid = [
        ("mpu1AccX", "MPU1 Acc X FFT"),
        ("mpu1AccY", "MPU1 Acc Y FFT"),
        ("mpu1AccZ", "MPU1 Acc Z FFT"),
        ("mpu1GyrX", "MPU1 Gyr X FFT"),
        ("mpu1GyrY", "MPU1 Gyr Y FFT"),
        ("mpu1GyrZ", "MPU1 Gyr Z FFT"),
        ("mpu2AccX", "MPU2 Acc X FFT"),
        ("mpu2AccY", "MPU2 Acc Y FFT"),
        ("mpu2AccZ", "MPU2 Acc Z FFT"),
        ("mpu2GyrX", "MPU2 Gyr X FFT"),
        ("mpu2GyrY", "MPU2 Gyr Y FFT"),
        ("mpu2GyrZ", "MPU2 Gyr Z FFT"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.patch.set_facecolor(BG_COLOR)

    for plotIndex, (signalPrefix, title) in enumerate(signalGrid):
        rowIndex = plotIndex // 3
        colIndex = plotIndex % 3
        ax = axes[rowIndex, colIndex]
        _styleAx(ax)
        plotAxisFftAx(
            ax,
            frequencyAxis,
            freqSignals[f"{signalPrefix}Spectrum"],
            freqSignals[f"{signalPrefix}FundamentalHz"],
            freqSignals[f"{signalPrefix}FundamentalMag"],
            plotMaxHz,
            title,
            rowIndex == 3,
            colIndex == 0,
        )

    fig.tight_layout(pad=1.0)
    plt.savefig(savePath, facecolor=BG_COLOR)
    plt.close(fig)
