import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG_COLOR = [0.08, 0.03, 0.03]
TEXT_COLOR = [0.98, 0.95, 0.82]
CLOSE_COLOR = [0.98, 0.98, 0.98]
GRID_COLOR = [0.45, 0.40, 0.35]


# Applies the shared axis styling.
def _styleAxes(ax):
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
def _styleLegend(legend):
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    frame = legend.get_frame()
    frame.set_facecolor(BG_COLOR)
    frame.set_edgecolor(GRID_COLOR)


# Plots the current time-domain magnitude and temperature-average signals.
def plotRaw(signalData, savePath):
    timeSignals = signalData["timeSignals"]
    sample = timeSignals["sample"]
    mpu1AccMag = timeSignals["mpu1AccMag"]
    mpu2AccMag = timeSignals["mpu2AccMag"]
    mpu1GyrMag = timeSignals["mpu1GyrMag"]
    mpu2GyrMag = timeSignals["mpu2GyrMag"]
    ds18b20OneAvgSample = timeSignals["ds18b20OneAvgSample"]
    ds18b20OneAvg = timeSignals["ds18b20OneAvg"]
    ds18b20TwoAvg = timeSignals["ds18b20TwoAvg"]

    samplePlot = sample[:100]
    mpu1AccMagPlot = mpu1AccMag[:100]
    mpu2AccMagPlot = mpu2AccMag[:100]
    mpu1GyrMagPlot = mpu1GyrMag[:100]
    mpu2GyrMagPlot = mpu2GyrMag[:100]
    tempSamplePlot = ds18b20OneAvgSample[:100]
    ds18b20OneAvgPlot = ds18b20OneAvg[:100]
    ds18b20TwoAvgPlot = ds18b20TwoAvg[:100]

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(BG_COLOR)
    grid = fig.add_gridspec(2, 2)
    accAx = fig.add_subplot(grid[0, 0])
    gyrAx = fig.add_subplot(grid[0, 1])
    tempAx = fig.add_subplot(grid[1, :])

    _styleAxes(accAx)
    _styleAxes(gyrAx)
    _styleAxes(tempAx)

    accAx.plot(
        samplePlot,
        mpu1AccMagPlot,
        color=CLOSE_COLOR,
        linewidth=1.2,
        label="mpu1AccMag",
    )
    accAx.plot(
        samplePlot,
        mpu2AccMagPlot,
        color="deepskyblue",
        linewidth=1.2,
        label="mpu2AccMag",
    )
    accAx.set_title("accel magnitude vs sample", color=TEXT_COLOR, pad=8)
    accAx.set_xlabel("sample", color=TEXT_COLOR)
    accAx.set_ylabel("accel mag", color=TEXT_COLOR)
    accLegend = accAx.legend(loc="best", frameon=True, fontsize=8)
    _styleLegend(accLegend)

    gyrAx.plot(
        samplePlot,
        mpu1GyrMagPlot,
        color=CLOSE_COLOR,
        linewidth=1.2,
        label="mpu1GyrMag",
    )
    gyrAx.plot(
        samplePlot,
        mpu2GyrMagPlot,
        color="deepskyblue",
        linewidth=1.2,
        label="mpu2GyrMag",
    )
    gyrAx.set_title("gyro magnitude vs sample", color=TEXT_COLOR, pad=8)
    gyrAx.set_xlabel("sample", color=TEXT_COLOR)
    gyrAx.set_ylabel("gyro mag", color=TEXT_COLOR)
    gyrLegend = gyrAx.legend(loc="best", frameon=True, fontsize=8)
    _styleLegend(gyrLegend)

    tempAx.plot(
        tempSamplePlot,
        ds18b20OneAvgPlot,
        color=CLOSE_COLOR,
        linewidth=1.2,
        label="ds18b20OneAvg",
    )
    tempAx.plot(
        tempSamplePlot,
        ds18b20TwoAvgPlot,
        color="deepskyblue",
        linewidth=1.2,
        label="ds18b20TwoAvg",
    )
    tempAx.set_title("temperature average vs sample", color=TEXT_COLOR, pad=8)
    tempAx.set_xlabel("sample", color=TEXT_COLOR)
    tempAx.set_ylabel("temperature", color=TEXT_COLOR)
    tempLegend = tempAx.legend(loc="best", frameon=True, fontsize=8)
    _styleLegend(tempLegend)

    fig.tight_layout(pad=0.8)
    plt.savefig(savePath, facecolor=BG_COLOR)
    plt.close(fig)
