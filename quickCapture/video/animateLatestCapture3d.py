import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "quickCapture/video/.mplconfig"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd


windowSamples = 2400
frameStep = 4
trailSamples = 120
smoothWindowSamples = 15
capturePattern = "capture_*.csv"
plotDirPath = "quickCapture/video/plots"
gifPathName = "latestCaptureCombined3dSmoothLong.gif"


# Returns the newest quick capture CSV path in the quickCapture tree.
def latestCapturePath():
    capturePaths = sorted(
        Path("quickCapture").rglob(capturePattern),
        key=lambda path: path.stat().st_mtime,
    )
    return capturePaths[-1]


# Returns the selected small sample window from the capture.
def captureWindow(dataFrame, windowSamples):
    windowFrame = dataFrame.iloc[:windowSamples].reset_index(drop=True)
    return windowFrame


# Applies a moving average to each selected axis.
def smoothAxes(windowFrame, axisKeys, smoothWindowSamples):
    smoothFrame = windowFrame[axisKeys].astype("float64").rolling(
        window=smoothWindowSamples,
        min_periods=1,
        center=True,
    ).mean()
    return smoothFrame


# Returns one centred and scaled 3-axis signal block.
def normAxes(axesFrame):
    centredFrame = axesFrame - axesFrame.mean()
    scale = centredFrame.abs().max()
    scaledFrame = centredFrame.divide(scale.replace(0.0, 1.0), axis=1)
    return scaledFrame


# Returns the combined MPU plot data for animation.
def mpuPlotData(windowFrame, mpuPrefix, smoothWindowSamples):
    accKeys = [
        f"{mpuPrefix}AccX",
        f"{mpuPrefix}AccY",
        f"{mpuPrefix}AccZ",
    ]
    gyrKeys = [
        f"{mpuPrefix}GyrX",
        f"{mpuPrefix}GyrY",
        f"{mpuPrefix}GyrZ",
    ]
    accFrame = smoothAxes(windowFrame, accKeys, smoothWindowSamples)
    gyrFrame = smoothAxes(windowFrame, gyrKeys, smoothWindowSamples)
    plotData = {
        "acc": normAxes(accFrame),
        "gyr": normAxes(gyrFrame),
    }
    return plotData


# Applies the shared 3D axis layout.
def styleAxis(ax, title):
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)


# Updates the current animation frame for one MPU subplot.
def updateMpuFrame(ax, plotData, frameIndex, trailSamples, title):
    ax.cla()
    styleAxis(ax, title)
    startIndex = max(0, frameIndex - trailSamples)
    accFrame = plotData["acc"].iloc[startIndex:frameIndex + 1]
    gyrFrame = plotData["gyr"].iloc[startIndex:frameIndex + 1]

    ax.plot(
        accFrame.iloc[:, 0],
        accFrame.iloc[:, 1],
        accFrame.iloc[:, 2],
        color="tab:blue",
        linewidth=1.6,
        label="acc",
    )
    ax.plot(
        gyrFrame.iloc[:, 0],
        gyrFrame.iloc[:, 1],
        gyrFrame.iloc[:, 2],
        color="tab:red",
        linewidth=1.2,
        label="gyro",
    )
    ax.scatter(
        [accFrame.iloc[-1, 0]],
        [accFrame.iloc[-1, 1]],
        [accFrame.iloc[-1, 2]],
        color="tab:blue",
        s=20,
    )
    ax.scatter(
        [gyrFrame.iloc[-1, 0]],
        [gyrFrame.iloc[-1, 1]],
        [gyrFrame.iloc[-1, 2]],
        color="tab:red",
        s=20,
    )
    ax.legend(loc="upper right")


# Saves one combined acc and gyro 3D animation for both MPUs.
def saveAnimation(
    windowFrame,
    latestPath,
    gifPath,
    frameStep,
    trailSamples,
    smoothWindowSamples,
):
    mpuOneData = mpuPlotData(
        windowFrame,
        "mpu1",
        smoothWindowSamples,
    )
    mpuTwoData = mpuPlotData(
        windowFrame,
        "mpu2",
        smoothWindowSamples,
    )
    frameIndexes = list(range(1, windowFrame.shape[0], frameStep))

    fig = plt.figure(figsize=(12, 6))
    axOne = fig.add_subplot(1, 2, 1, projection="3d")
    axTwo = fig.add_subplot(1, 2, 2, projection="3d")

    def draw(frameIndex):
        updateMpuFrame(
            axOne,
            mpuOneData,
            frameIndex,
            trailSamples,
            "MPU1 Acc + Gyro",
        )
        updateMpuFrame(
            axTwo,
            mpuTwoData,
            frameIndex,
            trailSamples,
            "MPU2 Acc + Gyro",
        )
        fig.suptitle(
            f"{latestPath.name} sample {frameIndex + 1}"
        )
        return []

    animation = FuncAnimation(
        fig,
        draw,
        frames=frameIndexes,
        interval=50,
        blit=False,
    )
    animation.save(gifPath, writer=PillowWriter(fps=20))
    plt.close(fig)


# Loads the newest capture, combines acc and gyro, and saves the GIF.
def main():
    latestPath = latestCapturePath()
    dataFrame = pd.read_csv(latestPath)
    windowFrame = captureWindow(dataFrame, windowSamples)
    plotDir = Path(plotDirPath)
    plotDir.mkdir(parents=True, exist_ok=True)
    gifPath = plotDir / gifPathName
    saveAnimation(
        windowFrame,
        latestPath,
        gifPath,
        frameStep,
        trailSamples,
        smoothWindowSamples,
    )
    print("capturePath:", latestPath)
    print("gifPath:", gifPath)
    print("sampleCount:", windowFrame.shape[0])
    print("smoothWindowSamples:", smoothWindowSamples)


if __name__ == "__main__":
    main()
