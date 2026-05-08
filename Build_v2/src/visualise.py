################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import sys

import matplotlib.pyplot as plt

import data
import signals

################################################################################
# variables/constants                                                          #
################################################################################
# visualise.py is an offline checking tool.
# It saves time-domain and FFT charts for one CSV recording.
buildDir = Path(__file__).resolve().parents[1]
trainDir = buildDir / "data" / "training" / "main"
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
    # Use a command line CSV if given.
    # Otherwise fall back to a default training file for quick testing.
    if len(sys.argv) > 1:
        csvPath = Path(sys.argv[1])
    else:
        csvPath = trainDir / "good1.csv"

    # Load and clean the recording, then calculate the same time and FFT signals
    # used by the training code.
    df = data.readCsv(csvPath)
    df = data.cleanFrame(df)
    timeData = signals.timeSignals(df, sampleRate)
    freqData = signals.fftSignals(
        timeData,
        sampleRate,
        signals.fftConfig,
    )

    # Save charts in a folder named after the CSV file.
    saveDir = outDir / csvPath.stem
    saveDir.mkdir(parents=True, exist_ok=True)

    # First chart: raw time-domain signal for all 12 sensor axes.
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.patch.set_facecolor(bgColor)
    for index, axisLabel in enumerate(signals.axisLabels):
        # Place each axis into a 4 x 3 grid.
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

    # Second chart: FFT spectrum for all 12 sensor axes.
    # Each subplot also marks the detected fundamental and BPFO/BPFI bands.
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.patch.set_facecolor(bgColor)
    for index, axisLabel in enumerate(signals.axisLabels):
        # Pull the pre-calculated spectrum and band values from freqData.
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
