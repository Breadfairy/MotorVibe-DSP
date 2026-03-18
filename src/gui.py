import charting


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
    print("featureVector:", mlData["featureVector"])
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
def printLiveSum(sensorColumns, signalData):
    print("sensorColumns:", sensorColumns)
    print("bufferMeta:", signalData["bufferMeta"])
    print("signalMeta:", signalData["signalMeta"])
    printSigSum(signalData)
    printMlSum(signalData)


# Renders one sensor exploration chart image for one MPU.
def plotSensor(signalData, sensorNumber, savePath, plotSeconds):
    charting.plotSensor(signalData, sensorNumber, savePath, plotSeconds)


# Renders one 12-panel per-axis FFT exploration chart image.
def plotAxisFftGrid(signalData, savePath):
    charting.plotAxisFftGrid(signalData, savePath)
