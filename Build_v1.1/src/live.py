################################################################################
# Imports                                                                      #
################################################################################
import math
import struct
import time

import pandas as pd
import serial

import ml_core
import signals_core

################################################################################
# variables/constants                                                          #
################################################################################
port = "COM5"
baudRate = 1000000
timeout = 1.0
sampRate = 1000.0
winSecs = 1.0
tempWinSecs = 0.01
bufferSampleCount = 32
recordFormat = "<I12hf"
accelScale = 16384.0
gyroScale = 131.0
readyPrefix = "Sample struct size (bytes):"
startCommand = b"START\n"
modelPath = "outputs/models/simpleMotorClassifier.pth"
alertProb = 0.60
recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount
winRows = int(sampRate * winSecs)

################################################################################
# helpers                                                                      #
################################################################################


# Renders one simple live text block in place.
def renderBlock(lines):
    blockText = "\n".join(lines)
    print(f"\x1b[2J\x1b[H{blockText}", end="", flush=True)


# Waits for the firmware text handshake before binary streaming starts.
def waitForReady(link):
    while True:
        deviceLine = link.readline().decode("ascii", errors="ignore").strip()
        if len(deviceLine) == 0:
            continue
        print("device:", deviceLine)
        if deviceLine.startswith(readyPrefix):
            break
    link.write(startCommand)
    link.flush()


################################################################################
# main functions                                                               #
################################################################################


modelValue = ml_core.model(ml_core.hiddenSize)
modelValue = ml_core.loadModel(modelValue, modelPath)
rows = []
totalRows = 0
rowsSinceUpdate = 0
remBytes = b""
startTs = 0.0
firstTUsRaw = None

print(f"starting live monitor on {port}")
print(f"baudRate: {baudRate}")
print(f"sampleRate: {sampRate}")
print(f"modelPath: {modelPath}")
print("waiting for device ready...")

with serial.Serial(
    port=port,
    baudrate=baudRate,
    timeout=timeout,
) as link:
    waitForReady(link)
    startTs = time.perf_counter()

    while True:
        newBytes = link.read(bufferSize)
        if len(newBytes) == 0:
            continue

        remBytes += newBytes

        while len(remBytes) >= recordSize:
            packetBytes = remBytes[:recordSize]
            remBytes = remBytes[recordSize:]
            packetValues = struct.unpack(recordFormat, packetBytes)
            tUsRaw = packetValues[0]
            if firstTUsRaw is None:
                firstTUsRaw = tUsRaw
            tUs = (tUsRaw - firstTUsRaw) & 0xFFFFFFFF
            tempC = packetValues[13]
            if not math.isfinite(tempC):
                tempC = 0.0
            rowValue = [
                tUs,
                tUs / 1e6,
                packetValues[1] / accelScale,
                packetValues[2] / accelScale,
                packetValues[3] / accelScale,
                packetValues[4] / gyroScale,
                packetValues[5] / gyroScale,
                packetValues[6] / gyroScale,
                packetValues[7] / accelScale,
                packetValues[8] / accelScale,
                packetValues[9] / accelScale,
                packetValues[10] / gyroScale,
                packetValues[11] / gyroScale,
                packetValues[12] / gyroScale,
                tempC,
            ]
            rows.append(rowValue)
            totalRows += 1
            rowsSinceUpdate += 1

        if len(rows) < winRows:
            continue

        rows = rows[-winRows:]

        ########################################################################
        # live ml update                                                       #
        ########################################################################

        if rowsSinceUpdate < winRows:
            continue

        rowsSinceUpdate = 0
        dataFrame = pd.DataFrame(rows, columns=signals_core.signalCols)
        rawSignals = signals_core.rawArrays(dataFrame)
        timeSignals = signals_core.timeData(
            rawSignals,
            sampRate,
            tempWinSecs,
        )
        freqSignals = signals_core.freqData(
            rawSignals,
            sampRate,
            signals_core.fftConfig,
        )
        featureVector = ml_core.featureVector(timeSignals, freqSignals)
        featureData = ml_core.featureTensor(featureVector[None, :])
        probTensor = ml_core.runModel(modelValue, featureData)
        probData = ml_core.probDict(probTensor)
        topName = ml_core.topLabel(probTensor)
        topProb = probData[topName]
        healthyProb = 0.0
        if "healthy" in probData:
            healthyProb = probData["healthy"]
        faultProb = 1.0 - healthyProb
        liveState = "watch"
        statusText = "mixed condition signature"
        if topName == "healthy" and topProb >= alertProb:
            liveState = "healthy"
            statusText = "healthy signature dominant"
        if topName != "healthy" and topProb >= alertProb:
            liveState = "warning"
            statusText = (
                f"{ml_core.displayLabel(topName)} signature rising"
            )

        runSecs = time.perf_counter() - startTs
        avgRate = totalRows / runSecs
        sortedLabels = sorted(
            probData.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        lines = [
            "Live Monitor",
            "",
            f"elapsedSeconds: {runSecs:.2f}",
            f"windowSeconds: {winSecs}",
            f"windowRows: {winRows}",
            f"avgSampleRate: {avgRate:.1f}",
            "",
            f"status: {liveState}",
            f"message: {statusText}",
            f"topClass: {ml_core.displayLabel(topName)}",
            f"topProb: {topProb * 100.0:.1f}%",
            f"healthyProb: {healthyProb * 100.0:.1f}%",
            f"faultProb: {faultProb * 100.0:.1f}%",
            "",
            f"tempAvgMean: {timeSignals['tempAvg'].mean():.3f}",
            f"tempGradMean: {timeSignals['tempGrad'].mean():.3f}",
            f"acc1FundHz: {freqSignals['acc1FundHz']:.3f}",
            f"acc2FundHz: {freqSignals['acc2FundHz']:.3f}",
            f"acc1BpfoRms: {freqSignals['acc1BpfoRms']:.3f}",
            f"acc2BpfoRms: {freqSignals['acc2BpfoRms']:.3f}",
            "",
            "Class Probabilities",
        ]
        for labelName, labelProb in sortedLabels:
            lines.append(
                f"{ml_core.displayLabel(labelName)}: "
                f"{labelProb * 100.0:.1f}%"
            )
        renderBlock(lines)
