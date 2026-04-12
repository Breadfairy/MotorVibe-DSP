import charting
import matplotlib.pyplot as plt


# Renders one in-place terminal dashboard.
def renderLiveBlock(lines):
    blockText = "\n".join(lines)
    print(f"\x1b[2J\x1b[H{blockText}", end="", flush=True)


# Builds formatted feature name and value lines.
def featureLines(mlData):
    lines = []
    for featureName, featureValue in zip(
        mlData["featureNames"],
        mlData["featureVector"],
    ):
        lines.append(f"{featureName}: {featureValue:.3f}")
    return lines


# Prints a short summary of the current signal outputs.
def printSigSum(signalData):
    rawSignals = signalData["rawSignals"]
    timeSignals = signalData["timeSignals"]
    freqSignals = signalData["freqSignals"]
    print("sample[:5]:", rawSignals["sample"][:5])
    print("mpu1AccMag[:5]:", timeSignals["mpu1AccMag"][:5])
    print("mpu2AccMag[:5]:", timeSignals["mpu2AccMag"][:5])
    print("mpu1GyrMag[:5]:", timeSignals["mpu1GyrMag"][:5])
    print("mpu2GyrMag[:5]:", timeSignals["mpu2GyrMag"][:5])
    print("mpu1AccFundamentalHz:", freqSignals["mpu1AccFundamentalHz"])
    print("mpu1AccFundamentalMag:", freqSignals["mpu1AccFundamentalMag"])
    print("mpu1AccBpfoBandRms:", freqSignals["mpu1AccBpfoBandRms"])
    print("mpu1AccBpfiBandRms:", freqSignals["mpu1AccBpfiBandRms"])
    print("mpu2AccFundamentalHz:", freqSignals["mpu2AccFundamentalHz"])
    print("mpu2AccFundamentalMag:", freqSignals["mpu2AccFundamentalMag"])
    print("mpu2AccBpfoBandRms:", freqSignals["mpu2AccBpfoBandRms"])
    print("mpu2AccBpfiBandRms:", freqSignals["mpu2AccBpfiBandRms"])


# Prints a short summary of the current ML-ready outputs.
def printMlSum(signalData):
    mlData = signalData["mlData"]
    print("featureCount:", len(mlData["featureNames"]))
    print("featureVectorShape:", mlData["featureVector"].shape)
    for line in featureLines(mlData):
        print(line)
    print("failureNames:", mlData["failureNames"])
    if "predictedLabel" in mlData:
        print("predictedLabel:", mlData["predictedLabel"])
        print("probabilityDict:", mlData["probabilityDict"])


# Prints a short summary of the current training dataset and result.
def printTrainSum(trainingData):
    trainingSet = trainingData["trainingSet"]
    print("trainingFeatureShape:", trainingSet["featureMatrix"].shape)
    print("trainingLabelShape:", trainingSet["labelVector"].shape)
    print("trainingDatasetMeta:", trainingSet["datasetMeta"])
    print("modelPath:", trainingData["modelPath"])
    print("lossStart:", trainingData["lossHistory"][0])
    print("lossEnd:", trainingData["lossHistory"][-1])
    print("predictedLabel:", trainingData["predictedLabel"])
    print("probabilityDict:", trainingData["probabilityDict"])


# Prints the current CSV replay summary to the terminal.
def printCsvSum(csvPath, sensorColumns, signalData):
    print("csvPath:", csvPath)
    print("sensorColumns:", sensorColumns)
    print("bufferMeta:", signalData["bufferMeta"])
    print("signalMeta:", signalData["signalMeta"])
    print("bufferFrameHead:")
    print(signalData["bufferFrame"].head())
    printSigSum(signalData)
    printMlSum(signalData)


# Prints the current live sensing summary to the terminal.
def printLiveSum(sensorColumns, signalData, rateState):
    rawSignals = signalData["rawSignals"]
    timeSignals = signalData["timeSignals"]
    signalMeta = signalData["signalMeta"]
    bufferMeta = signalData["bufferMeta"]
    streamDiagnostics = signalData["streamDiagnostics"]
    measuredSampleRate = rateState["measuredSampleRate"]
    measuredRateText = "--"
    if measuredSampleRate is not None:
        measuredRateText = f"{measuredSampleRate:.1f}"
    goodBatchCount = "--"
    badBatchCount = "--"
    consecutiveBadBatchCount = "--"
    lastBadBatchMessage = "none"
    if streamDiagnostics is not None:
        goodBatchCount = streamDiagnostics["goodBatchCount"]
        badBatchCount = streamDiagnostics["badBatchCount"]
        consecutiveBadBatchCount = (
            streamDiagnostics["consecutiveBadBatchCount"]
        )
        if streamDiagnostics["lastBadBatchMessage"] is not None:
            lastBadBatchMessage = streamDiagnostics["lastBadBatchMessage"]
    lines = [
        "Main Live Monitor",
        "",
        f"sensorColumns: {sensorColumns}",
        f"bufferMeta: {bufferMeta}",
        f"signalMeta: {signalMeta}",
        f"measuredSampleRate: {measuredRateText}",
        f"goodBatchCount: {goodBatchCount}",
        f"badBatchCount: {badBatchCount}",
        f"consecutiveBadBatchCount: {consecutiveBadBatchCount}",
        f"lastBadBatch: {lastBadBatchMessage}",
        "",
        f"sampleTail: {rawSignals['sample'][-5:]}",
        f"mpu1AccMagMean: {timeSignals['mpu1AccMag'].mean():.3f}",
        f"mpu1AccMagMax: {timeSignals['mpu1AccMag'].max():.3f}",
        f"mpu1GyrMagMean: {timeSignals['mpu1GyrMag'].mean():.3f}",
        f"mpu2AccMagMean: {timeSignals['mpu2AccMag'].mean():.3f}",
        f"mpu2GyrMagMean: {timeSignals['mpu2GyrMag'].mean():.3f}",
        f"ds18b20AvgMean: {timeSignals['ds18b20Avg'].mean():.3f}",
    ]
    renderLiveBlock(lines)


# Builds one simple live acc magnitude monitor figure.
def liveAccMagFig(bufferSeconds):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(charting.BG_COLOR)
    charting._styleAx(ax)
    line, = ax.plot(
        [],
        [],
        color=charting.ACC_COLOR,
        linewidth=1.6,
    )
    ax.set_title("MPU6050 1 accMag live", color=charting.TEXT_COLOR)
    ax.set_xlabel("seconds trailing right", color=charting.TEXT_COLOR)
    ax.set_ylabel("magnitude", color=charting.TEXT_COLOR)
    ax.set_xlim(0.0, bufferSeconds)
    ax.invert_xaxis()
    ax.set_ylim(0.0, 30000.0)
    plt.show(block=False)
    livePlot = {
        "fig": fig,
        "ax": ax,
        "line": line,
        "bufferSeconds": bufferSeconds,
    }
    return livePlot


# Updates one simple live acc magnitude monitor figure.
def updateLiveAccMag(livePlot, signalData):
    sampleRate = signalData["signalMeta"]["sampleRate"]
    sample = signalData["rawSignals"]["sample"]
    accMag = signalData["timeSignals"]["mpu1AccMag"]
    trailSeconds = (sample[-1] - sample) / sampleRate
    livePlot["line"].set_data(trailSeconds, accMag)
    livePlot["ax"].set_xlim(0.0, livePlot["bufferSeconds"])
    livePlot["ax"].invert_xaxis()
    yMax = max(float(accMag.max()) * 1.8, 30000.0)
    livePlot["ax"].set_ylim(0.0, yMax)
    livePlot["fig"].canvas.draw_idle()
    livePlot["fig"].canvas.flush_events()
    plt.pause(0.001)


# Renders one sensor exploration chart image for one MPU.
def plotSensor(signalData, sensorNumber, savePath, plotSeconds):
    charting.plotSensor(signalData, sensorNumber, savePath, plotSeconds)


# Renders one 12-panel per-axis FFT exploration chart image.
def plotAxisFftGrid(signalData, savePath):
    charting.plotAxisFftGrid(signalData, savePath)
