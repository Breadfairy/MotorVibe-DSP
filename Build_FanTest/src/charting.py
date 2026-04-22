################################################################################
# Imports                                                                      #
################################################################################
import os
from pathlib import Path
import tempfile

cacheDir = Path(tempfile.gettempdir()) / "build_fantest_mpl"
cacheDir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cacheDir)
os.environ["XDG_CACHE_HOME"] = str(cacheDir)

import matplotlib
matplotlib.use(os.environ.get("BUILD_FANTEST_MPL_BACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np

import signals

################################################################################
# variables/constants                                                          #
################################################################################
bgColor = [0.08, 0.03, 0.03]
textColor = [0.98, 0.95, 0.82]
gridColor = [0.45, 0.40, 0.35]
lineColor = [0.98, 0.98, 0.98]
sumColor = "deepskyblue"
bpfoColor = "tomato"
bpfiColor = "seagreen"
plotMaxHz = {
    "sixaxis_1k": 500.0,
    "gyro_8k": 4000.0,
}
liveTimeYMax = {
    "sixaxis_1k": 500.0,
    "gyro_8k": 500.0,
}
liveMaxPoints = 1200

################################################################################
# helpers                                                                      #
################################################################################


# Applies the shared axis styling.
def styleAx(ax):
    ax.set_facecolor(bgColor)
    ax.tick_params(colors=textColor, labelsize=8)
    for i in ax.spines.values():
        i.set_color(gridColor)
    ax.grid(
        color=gridColor,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )


# Returns a flat list from the current subplot object.
def flatAxes(axes):
    return list(np.atleast_1d(axes).ravel())


# Returns the current subplot shape for one mode.
def gridShape(mode):
    if mode == "sixaxis_1k":
        return 2, 3
    return 1, 3


# Returns one decimated x and y pair for live plotting.
def livePoints(y, maxPoints):
    n = len(y)
    if n <= maxPoints:
        x = np.arange(n, dtype=np.float64)
        return x, y
    step = int(np.ceil(n / maxPoints))
    i = np.arange(0, n, step, dtype=np.int64)
    x = i.astype(np.float64)
    return x, y[i]


# Moves one span patch to the selected x range.
def moveBand(patch, lowHz, highHz):
    patch.set_x(lowHz)
    patch.set_width(highHz - lowHz)

################################################################################
# main functions                                                               #
################################################################################


# Saves one raw time-axis grid for the current signal block.
def plotTimeGrid(sig, savePath):
    rows, cols = gridShape(sig["mode"])
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7))
    fig.patch.set_facecolor(bgColor)
    for ax, i in zip(flatAxes(axes), sig["timeCols"]):
        styleAx(ax)
        ax.plot(
            sig["t_s"],
            sig[i],
            color=lineColor,
            linewidth=1.0,
        )
        ax.set_title(signals.axisLabel[i], color=textColor, pad=6)
        ax.set_xlabel("time s", color=textColor)
        ax.set_ylabel("value", color=textColor)
    fig.tight_layout(pad=1.0)
    plt.savefig(savePath, facecolor=bgColor)
    plt.close(fig)


# Saves one raw FFT grid for the current signal block.
def plotFftGrid(sig, savePath):
    rows, cols = gridShape(sig["mode"])
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7))
    fig.patch.set_facecolor(bgColor)
    maxHz = plotMaxHz[sig["mode"]]
    mask = sig["freqHz"] <= maxHz
    for ax, i in zip(flatAxes(axes), sig["fftCols"]):
        styleAx(ax)
        ax.plot(
            sig["freqHz"][mask],
            sig[f"{i}Fft"][mask],
            color=lineColor,
            linewidth=1.0,
        )
        ax.set_title(signals.axisLabel[i], color=textColor, pad=6)
        ax.set_xlabel("frequency hz", color=textColor)
        ax.set_ylabel("magnitude", color=textColor)
        ax.set_xlim(0.0, maxHz)
    fig.tight_layout(pad=1.0)
    plt.savefig(savePath, facecolor=bgColor)
    plt.close(fig)


# Saves one visual summary with time magnitude and summed FFT.
def plotSummary(sig, savePath):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(bgColor)
    styleAx(axes[0])
    styleAx(axes[1])
    axes[0].plot(
        sig["t_s"],
        sig["timeMag"],
        color=lineColor,
        linewidth=1.2,
    )
    axes[0].set_title("Time Magnitude", color=textColor)
    axes[0].set_xlabel("time s", color=textColor)
    axes[0].set_ylabel("magnitude", color=textColor)
    maxHz = plotMaxHz[sig["mode"]]
    mask = sig["freqHz"] <= maxHz
    axes[1].plot(
        sig["freqHz"][mask],
        sig["sumFft"][mask],
        color=sumColor,
        linewidth=1.1,
    )
    axes[1].axvspan(
        sig["bpfoBand"][0],
        sig["bpfoBand"][1],
        color=bpfoColor,
        alpha=0.12,
    )
    axes[1].axvspan(
        sig["bpfiBand"][0],
        sig["bpfiBand"][1],
        color=bpfiColor,
        alpha=0.10,
    )
    axes[1].scatter(
        [sig["fundHz"]],
        [sig["fundMag"]],
        color=lineColor,
        s=20,
        zorder=3,
    )
    axes[1].set_title("Summed FFT", color=textColor)
    axes[1].set_xlabel("frequency hz", color=textColor)
    axes[1].set_ylabel("magnitude", color=textColor)
    axes[1].set_xlim(0.0, maxHz)
    fig.tight_layout(pad=0.8)
    plt.savefig(savePath, facecolor=bgColor)
    plt.close(fig)


# Builds one live visual figure for the current stream mode.
def buildLiveFig(mode, plotRows):
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(bgColor)
    styleAx(axes[0])
    styleAx(axes[1])
    axes[0].set_title("Time Magnitude", color=textColor)
    axes[0].set_xlabel("samples", color=textColor)
    axes[0].set_ylabel("magnitude", color=textColor)
    axes[1].set_title("Summed FFT", color=textColor)
    axes[1].set_xlabel("frequency hz", color=textColor)
    axes[1].set_ylabel("magnitude", color=textColor)
    axes[0].set_xlim(0.0, float(plotRows - 1))
    axes[0].set_ylim(0.0, liveTimeYMax[mode])
    axes[1].set_xlim(0.0, plotMaxHz[mode])
    timeLine, = axes[0].plot([], [], color=lineColor, linewidth=1.2)
    fftLine, = axes[1].plot([], [], color=sumColor, linewidth=1.1)
    fundMark = axes[1].scatter([], [], color=lineColor, s=20, zorder=3)
    bpfoPatch = axes[1].axvspan(0.0, 0.0, color=bpfoColor, alpha=0.12)
    bpfiPatch = axes[1].axvspan(0.0, 0.0, color=bpfiColor, alpha=0.10)
    fig.tight_layout(pad=0.8)
    plt.show(block=False)
    return {
        "fig": fig,
        "timeAx": axes[0],
        "fftAx": axes[1],
        "timeLine": timeLine,
        "fftLine": fftLine,
        "fundMark": fundMark,
        "bpfoPatch": bpfoPatch,
        "bpfiPatch": bpfiPatch,
        "mode": mode,
    }


# Updates one live visual figure from the current signal block.
def updateLiveFig(liveFig, sig):
    fftAx = liveFig["fftAx"]
    timeLine = liveFig["timeLine"]
    fftLine = liveFig["fftLine"]
    fundMark = liveFig["fundMark"]
    mode = liveFig["mode"]
    maxHz = plotMaxHz[mode]
    mask = sig["freqHz"] <= maxHz
    x0, y0 = livePoints(sig["timeMag"], liveMaxPoints)
    x1, y1 = livePoints(sig["sumFft"][mask], liveMaxPoints)
    timeLine.set_data(x0, y0)
    fftLine.set_data(sig["freqHz"][mask][x1.astype(np.int64)], y1)
    y2 = float(np.max(sig["sumFft"][mask]))
    if y2 <= 0.0:
        y2 = 0.1
    fftAx.set_ylim(0.0, y2 * 1.1)
    fundMark.set_offsets([[sig["fundHz"], sig["fundMag"]]])
    moveBand(liveFig["bpfoPatch"], sig["bpfoBand"][0], sig["bpfoBand"][1])
    moveBand(liveFig["bpfiPatch"], sig["bpfiBand"][0], sig["bpfiBand"][1])
    fftAx.set_title(
        f"Summed FFT {sig['fundHz']:.2f} Hz",
        color=textColor,
    )
    liveFig["fig"].canvas.draw_idle()
    liveFig["fig"].canvas.flush_events()
    plt.pause(0.001)
