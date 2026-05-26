################################################################################
# Imports                                                                      #
################################################################################
import os
from pathlib import Path
import sys

buildDir = Path(__file__).resolve().parents[1]
mplConfigDir = buildDir / "outputs" / ".matplotlib"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(mplConfigDir))

import matplotlib.pyplot as plt

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
# Offline charts for one CSV recording.
outDir = buildDir / "outputs" / "charts"
sampleRate = 1000.0
bgColor = [0.08, 0.03, 0.03]
textColor = [0.98, 0.95, 0.82]
gridColor = [0.45, 0.40, 0.35]
lineColor = [0.98, 0.98, 0.98]
bpfoColor = "tomato"
bpfiColor = "seagreen"
fundColor = "deepskyblue"
plotMaxHz = 500.0

# Human-readable names for saved signal charts.
axisLabels = [
    ("ax1", "MPU1 Acc X"),
    ("ay1", "MPU1 Acc Y"),
    ("az1", "MPU1 Acc Z"),
    ("gx1", "MPU1 Gyr X"),
    ("gy1", "MPU1 Gyr Y"),
    ("gz1", "MPU1 Gyr Z"),
    ("ax2", "MPU2 Acc X"),
    ("ay2", "MPU2 Acc Y"),
    ("az2", "MPU2 Acc Z"),
    ("gx2", "MPU2 Gyr X"),
    ("gy2", "MPU2 Gyr Y"),
    ("gz2", "MPU2 Gyr Z"),
]

################################################################################
# helpers                                                                      #
################################################################################


# Applies the same dark chart style used by the live plot.
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

################################################################################
# main functions                                                               #
################################################################################


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 visualise.py path/to/recording.csv")
    csvPath = Path(sys.argv[1])

    df = data.readCsv(csvPath)
    df = data.cleanFrame(df)
    timeData = signals.plotTime(df, sampleRate)
    freqData = signals.plotFreq(
        timeData,
        sampleRate,
        signals.fftConfig,
    )

    saveDir = outDir / csvPath.stem
    saveDir.mkdir(parents=True, exist_ok=True)

    # Raw time-domain axes.
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.patch.set_facecolor(bgColor)
    for index, axisLabel in enumerate(axisLabels):
        row = index // 3
        col = index % 3
        ax = axes[row, col]
        styleAx(ax)
        ax.plot(
            timeData["t_s"],
            timeData[axisLabel[0]],
            color=lineColor,
            linewidth=1.0,
        )
        ax.set_title(axisLabel[1], color=textColor, pad=6, fontsize=9)
        ax.set_xlabel("time s", color=textColor)
        ax.set_ylabel("value", color=textColor)
    timePath = saveDir / "timeGrid.png"
    fig.tight_layout(pad=1.0)
    plt.savefig(timePath, facecolor=bgColor)
    plt.close(fig)

    # Raw-axis FFTs with order bands.
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.patch.set_facecolor(bgColor)
    for index, axisLabel in enumerate(axisLabels):
        row = index // 3
        col = index % 3
        ax = axes[row, col]
        styleAx(ax)
        mask = freqData["freqAxis"] <= plotMaxHz
        hz = freqData["freqAxis"][mask]
        mag = freqData[f"{axisLabel[0]}Spectrum"][mask]
        fundHz = freqData[f"{axisLabel[0]}FundHz"]
        fundMag = freqData[f"{axisLabel[0]}FundMag"]
        bpfo = freqData[f"{axisLabel[0]}BpfoBand"]
        bpfi = freqData[f"{axisLabel[0]}BpfiBand"]
        ax.plot(hz, mag, color=lineColor, linewidth=1.0)
        ax.axvspan(bpfo[0], bpfo[1], color=bpfoColor, alpha=0.12)
        ax.axvspan(bpfi[0], bpfi[1], color=bpfiColor, alpha=0.10)
        ax.scatter([fundHz], [fundMag], color=fundColor, s=16, zorder=3)
        ax.set_title(
            f"{axisLabel[1]} {fundHz:.2f} Hz",
            color=textColor,
            pad=6,
            fontsize=9,
        )
        ax.set_xlabel("frequency hz", color=textColor)
        ax.set_ylabel("magnitude", color=textColor)
        ax.set_xlim(0.0, plotMaxHz)
    fftPath = saveDir / "fftGrid.png"
    fig.tight_layout(pad=1.0)
    plt.savefig(fftPath, facecolor=bgColor)
    plt.close(fig)

    print("timePath:", timePath)
    print("fftPath:", fftPath)


if __name__ == "__main__":
    main()
