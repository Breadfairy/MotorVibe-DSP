import charting


# Prints a short summary of the current signal outputs.
def printSignalSummary(signalData):
    rawSignals = signalData["rawSignals"]
    timeSignals = signalData["timeSignals"]
    freqSignals = signalData["freqSignals"]
    print("sample[:5]:", rawSignals["sample"][:5])
    print("mpu1AccMag[:5]:", timeSignals["mpu1AccMag"][:5])
    print("mpu2AccMag[:5]:", timeSignals["mpu2AccMag"][:5])
    print("mpu1GyrMag[:5]:", timeSignals["mpu1GyrMag"][:5])
    print("mpu2GyrMag[:5]:", timeSignals["mpu2GyrMag"][:5])
    print("ds18b20OneAvg[:5]:", timeSignals["ds18b20OneAvg"][:5])
    print("ds18b20OneGrad[:5]:", timeSignals["ds18b20OneGrad"][:5])
    print("mpu1AccMagDomFreq:", freqSignals["mpu1AccMagDomFreq"])
    print("mpu1AccMagDomMag:", freqSignals["mpu1AccMagDomMag"])
    print(
        "mpu1AccAxisPowerEnergy1_A:",
        freqSignals["mpu1AccAxisPowerEnergy1_A"],
    )
    print(
        "mpu1AccAxisPowerEnergy2_A:",
        freqSignals["mpu1AccAxisPowerEnergy2_A"],
    )
    print(
        "mpu1AccAxisPowerEnergy3_A:",
        freqSignals["mpu1AccAxisPowerEnergy3_A"],
    )


# Prints a short summary of the current ML-ready outputs.
def printMLSummary(signalData):
    mlData = signalData["mlData"]
    print("featureCount:", len(mlData["featureNames"]))
    print("featureVectorShape:", mlData["featureVector"].shape)
    print("featureVector[:5]:", mlData["featureVector"][:5])
    print("failureNames:", mlData["failureNames"])
    if "predictedLabel" in mlData:
        print("predictedLabel:", mlData["predictedLabel"])
        print("probabilityDict:", mlData["probabilityDict"])


# Prints a short summary of the current training dataset and result.
def printTrainingSummary(trainingData):
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
def printCSVSummary(csvPath, sensorColumns, signalData):
    print("csvPath:", csvPath)
    print("sensorColumns:", sensorColumns)
    print("bufferMeta:", signalData["bufferMeta"])
    print("signalMeta:", signalData["signalMeta"])
    print("bufferFrameHead:")
    print(signalData["bufferFrame"].head())
    printSignalSummary(signalData)
    printMLSummary(signalData)


# Prints the current live sensing summary to the terminal.
def printLiveSummary(sensorColumns, signalData):
    print("sensorColumns:", sensorColumns)
    print("bufferMeta:", signalData["bufferMeta"])
    print("signalMeta:", signalData["signalMeta"])
    printSignalSummary(signalData)
    printMLSummary(signalData)


# Renders the current raw chart image.
def plotRaw(signalData, savePath):
    charting.plotRaw(signalData, savePath)
