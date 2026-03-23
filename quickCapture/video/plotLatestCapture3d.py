from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


windowSamples = 500
capturePattern = "capture_*.csv"
plotDirPath = "quickCapture/video/plots"


# Returns the newest quick capture CSV path in the quickCapture tree.
def latestCapturePath():
    capturePaths = sorted(
        Path("quickCapture").rglob(capturePattern),
        key=lambda path: path.stat().st_mtime,
    )
    return capturePaths[-1]


# Returns the selected small sample window from the capture.
def captureWindow(dataFrame, windowSamples):
    return dataFrame.iloc[:windowSamples].reset_index(drop=True)


# Saves one 3D accelerometer trace figure for both MPUs.
def save3dPlot(windowFrame, latestPath, plotPath):
    fig = plt.figure(figsize=(12, 6))
    axOne = fig.add_subplot(1, 2, 1, projection="3d")
    axTwo = fig.add_subplot(1, 2, 2, projection="3d")

    axOne.plot(
        windowFrame["mpu1AccX"],
        windowFrame["mpu1AccY"],
        windowFrame["mpu1AccZ"],
        linewidth=1.2,
        color="tab:blue",
    )
    axOne.set_title("MPU1 Acc 3D")
    axOne.set_xlabel("accX")
    axOne.set_ylabel("accY")
    axOne.set_zlabel("accZ")

    axTwo.plot(
        windowFrame["mpu2AccX"],
        windowFrame["mpu2AccY"],
        windowFrame["mpu2AccZ"],
        linewidth=1.2,
        color="tab:orange",
    )
    axTwo.set_title("MPU2 Acc 3D")
    axTwo.set_xlabel("accX")
    axTwo.set_ylabel("accY")
    axTwo.set_zlabel("accZ")

    fig.suptitle(f"{latestPath.name} first {windowFrame.shape[0]} samples")
    fig.tight_layout()
    fig.savefig(plotPath, dpi=180)
    plt.close(fig)


# Loads the newest capture, slices a small window, and saves the plot.
def main():
    latestPath = latestCapturePath()
    dataFrame = pd.read_csv(latestPath)
    windowFrame = captureWindow(dataFrame, windowSamples)
    plotDir = Path(plotDirPath)
    plotDir.mkdir(parents=True, exist_ok=True)
    plotPath = plotDir / "latestCapture3d.png"
    save3dPlot(windowFrame, latestPath, plotPath)
    print("capturePath:", latestPath)
    print("plotPath:", plotPath)
    print("sampleCount:", windowFrame.shape[0])


if __name__ == "__main__":
    main()
