import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "quickCapture/video/.mplconfig"

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sampleRate = 1000
startSeconds = 1
windowSamples = 6000
frameStep = 6
trailSamples = 180
smoothWindowSamples = 15
fps = 24
capturePattern = "capture_*.csv"
plotDirPath = "quickCapture/video/plots"
mp4PathName = "latestCaptureLongSmooth.mp4"


# Returns the newest quick capture CSV path in the quickCapture tree.
def latestCapturePath():
    capturePaths = sorted(
        Path("quickCapture").rglob(capturePattern),
        key=lambda path: path.stat().st_mtime,
    )
    return capturePaths[-1]


# Returns the selected capture window after the chosen start offset.
def captureWindow(dataFrame, sampleRate, startSeconds, windowSamples):
    startSample = int(sampleRate * startSeconds)
    stopSample = startSample + windowSamples
    windowFrame = dataFrame.iloc[startSample:stopSample].reset_index(
        drop=True
    )
    return windowFrame, startSample


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


# Returns the smoothed and normalized data for one 3-axis block.
def axisPlotData(windowFrame, axisKeys, smoothWindowSamples):
    smoothFrame = smoothAxes(windowFrame, axisKeys, smoothWindowSamples)
    return normAxes(smoothFrame)


# Returns all subplot data blocks for the current window.
def plotData(windowFrame, smoothWindowSamples):
    data = {
        "mpu1Acc": axisPlotData(
            windowFrame,
            ["mpu1AccX", "mpu1AccY", "mpu1AccZ"],
            smoothWindowSamples,
        ),
        "mpu1Gyr": axisPlotData(
            windowFrame,
            ["mpu1GyrX", "mpu1GyrY", "mpu1GyrZ"],
            smoothWindowSamples,
        ),
        "mpu2Acc": axisPlotData(
            windowFrame,
            ["mpu2AccX", "mpu2AccY", "mpu2AccZ"],
            smoothWindowSamples,
        ),
        "mpu2Gyr": axisPlotData(
            windowFrame,
            ["mpu2GyrX", "mpu2GyrY", "mpu2GyrZ"],
            smoothWindowSamples,
        ),
    }
    return data


# Applies the shared 3D axis layout.
def styleAxis(ax, title):
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)


# Draws one 3-axis trail and head marker on the target subplot.
def drawTrail(ax, axesFrame, frameIndex, trailSamples, color, title):
    ax.cla()
    styleAxis(ax, title)
    startIndex = max(0, frameIndex - trailSamples)
    trailFrame = axesFrame.iloc[startIndex:frameIndex + 1]
    ax.plot(
        trailFrame.iloc[:, 0],
        trailFrame.iloc[:, 1],
        trailFrame.iloc[:, 2],
        color=color,
        linewidth=1.4,
    )
    ax.scatter(
        [trailFrame.iloc[-1, 0]],
        [trailFrame.iloc[-1, 1]],
        [trailFrame.iloc[-1, 2]],
        color=color,
        s=22,
    )


# Renders one full animation frame into an RGB array.
def frameImage(
    figure,
    axes,
    data,
    frameIndex,
    trailSamples,
    latestPath,
    absoluteSample,
):
    drawTrail(
        axes["mpu1Acc"],
        data["mpu1Acc"],
        frameIndex,
        trailSamples,
        "tab:blue",
        "MPU1 Acc",
    )
    drawTrail(
        axes["mpu1Gyr"],
        data["mpu1Gyr"],
        frameIndex,
        trailSamples,
        "tab:red",
        "MPU1 Gyro",
    )
    drawTrail(
        axes["mpu2Acc"],
        data["mpu2Acc"],
        frameIndex,
        trailSamples,
        "tab:green",
        "MPU2 Acc",
    )
    drawTrail(
        axes["mpu2Gyr"],
        data["mpu2Gyr"],
        frameIndex,
        trailSamples,
        "tab:orange",
        "MPU2 Gyro",
    )
    figure.suptitle(
        f"{latestPath.name} sample {absoluteSample + frameIndex + 1}"
    )
    figure.canvas.draw()
    frameArray = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
    return frameArray


# Saves one long smoothed MP4 from the newest capture window.
def saveMp4(
    windowFrame,
    latestPath,
    mp4Path,
    startSample,
    frameStep,
    trailSamples,
    smoothWindowSamples,
    fps,
):
    data = plotData(windowFrame, smoothWindowSamples)
    figure = plt.figure(figsize=(12, 10))
    axes = {
        "mpu1Acc": figure.add_subplot(2, 2, 1, projection="3d"),
        "mpu1Gyr": figure.add_subplot(2, 2, 2, projection="3d"),
        "mpu2Acc": figure.add_subplot(2, 2, 3, projection="3d"),
        "mpu2Gyr": figure.add_subplot(2, 2, 4, projection="3d"),
    }
    frameIndexes = list(range(1, windowFrame.shape[0], frameStep))

    with imageio.get_writer(
        mp4Path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    ) as writer:
        for frameIndex in frameIndexes:
            frameArray = frameImage(
                figure,
                axes,
                data,
                frameIndex,
                trailSamples,
                latestPath,
                startSample,
            )
            writer.append_data(frameArray)

    plt.close(figure)
    return len(frameIndexes)


# Loads the newest capture and saves the configured MP4.
def main():
    latestPath = latestCapturePath()
    dataFrame = pd.read_csv(latestPath)
    windowFrame, startSample = captureWindow(
        dataFrame,
        sampleRate,
        startSeconds,
        windowSamples,
    )
    plotDir = Path(plotDirPath)
    plotDir.mkdir(parents=True, exist_ok=True)
    mp4Path = plotDir / mp4PathName
    frameCount = saveMp4(
        windowFrame,
        latestPath,
        mp4Path,
        startSample,
        frameStep,
        trailSamples,
        smoothWindowSamples,
        fps,
    )
    print("capturePath:", latestPath)
    print("mp4Path:", mp4Path)
    print("startSample:", startSample)
    print("sampleCount:", windowFrame.shape[0])
    print("smoothWindowSamples:", smoothWindowSamples)
    print("frameCount:", frameCount)


if __name__ == "__main__":
    main()
