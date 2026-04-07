import numpy as np
import pandas as pd
from scipy import fft

import buffer


# Converts named sensor columns into NumPy float64 arrays.
def rawSigs(bufferData):
    rawSignals = {}
    for sensorColumn, sensorArray in bufferData["bufferArrays"].items():
        rawSignals[sensorColumn] = sensorArray.astype(np.float64)
    return rawSignals


# Slices the latest analysis-period view from the current buffer data.
def periodBuf(bufferData, sampleRate, periodSeconds):
    periodSampleCount = int(sampleRate * periodSeconds)
    periodFrame = bufferData["bufferFrame"].iloc[-periodSampleCount:]
    periodFrame = periodFrame.reset_index(drop=True)
    sensorColumns = list(bufferData["bufferArrays"].keys())
    periodBufferData = {
        "bufferFrame": periodFrame,
        "bufferArrays": buffer.bufArrs(periodFrame, sensorColumns),
        "bufferMeta": bufferData["bufferMeta"],
    }
    return periodBufferData


# Calculates the magnitude of a 3-axis signal.
def mag3(axisX, axisY, axisZ):
    magnitude = np.sqrt(axisX**2 + axisY**2 + axisZ**2)
    return magnitude


# Calculates a simple moving average on a 1D array.
def movAvg(signal, windowSize):
    kernel = np.ones(windowSize, dtype=np.float64) / windowSize
    averagedSignal = np.convolve(signal, kernel, mode="valid")
    return averagedSignal


# Builds a matching sample axis for a reduced-length signal.
def trimSamp(sample, reducedSize):
    reducedSample = sample[-reducedSize:]
    return reducedSample


# Calculates the first derivative of a 1D array against time.
def gradSig(signal, sampleRate):
    gradient = np.gradient(signal) * sampleRate
    return gradient


# Builds one DC-removed FFT magnitude spectrum.
def buildSpectrum(signal, sampleRate):
    sampleCount = signal.size
    centeredSignal = signal - np.mean(signal)
    spectrum = fft.rfft(centeredSignal)
    magnitude = (2.0 / sampleCount) * np.abs(spectrum)
    frequencyAxis = fft.rfftfreq(sampleCount, d=1.0 / sampleRate)
    return frequencyAxis, magnitude


# Finds one dominant peak inside the selected frequency window.
def fundamentalPeak(frequencyAxis, magnitude, minFundHz, maxFundHz):
    fundMask = (
        (frequencyAxis >= minFundHz)
        & (frequencyAxis <= maxFundHz)
    )
    fundFrequencies = frequencyAxis[fundMask]
    fundMagnitudes = magnitude[fundMask]
    peakIndex = int(np.argmax(fundMagnitudes))
    peakFrequency = float(fundFrequencies[peakIndex])
    peakMagnitude = float(fundMagnitudes[peakIndex])
    return peakFrequency, peakMagnitude


# Builds one RMS magnitude for one selected band.
def bandRms(frequencyAxis, magnitude, bandLowHz, bandHighHz):
    bandMask = (
        (frequencyAxis >= bandLowHz)
        & (frequencyAxis <= bandHighHz)
    )
    bandMagnitudes = magnitude[bandMask]
    bandRmsValue = float(np.sqrt(np.mean(bandMagnitudes**2)))
    return bandRmsValue


# Builds one order band with a symmetric tolerance.
def orderBand(fundamentalHz, lowOrder, highOrder, toleranceFraction):
    bandLowHz = (fundamentalHz * lowOrder) * (
        1.0 - toleranceFraction
    )
    bandHighHz = (fundamentalHz * highOrder) * (
        1.0 + toleranceFraction
    )
    return bandLowHz, bandHighHz


# Builds FFT spectra and peak metrics for the X, Y, and Z axes.
def axisFftSigs(axisX, axisY, axisZ, sampleRate, fftConfig):
    axisSignals = {}
    for axisName, axisSignal in [
        ("X", axisX),
        ("Y", axisY),
        ("Z", axisZ),
    ]:
        frequencyAxis, axisSpectrum = buildSpectrum(
            axisSignal,
            sampleRate,
        )
        fundamentalHz, fundamentalMag = fundamentalPeak(
            frequencyAxis,
            axisSpectrum,
            fftConfig["minFundHz"],
            fftConfig["maxFundHz"],
        )
        axisSignals[axisName] = {
            "frequencyAxis": frequencyAxis,
            "spectrum": axisSpectrum,
            "fundamentalHz": fundamentalHz,
            "fundamentalMag": fundamentalMag,
        }
    return axisSignals


# Combines the three axis spectra into one plot-level spectrum summary.
def combAxisFftSigs(axisSignals, fftConfig):
    frequencyAxis = axisSignals["X"]["frequencyAxis"]
    combinedSpectrum = np.sqrt(
        axisSignals["X"]["spectrum"]**2
        + axisSignals["Y"]["spectrum"]**2
        + axisSignals["Z"]["spectrum"]**2
    )
    fundamentalHz, fundamentalMag = fundamentalPeak(
        frequencyAxis,
        combinedSpectrum,
        fftConfig["minFundHz"],
        fftConfig["maxFundHz"],
    )
    bpfoBand = orderBand(
        fundamentalHz,
        fftConfig["bpfoLowOrder"],
        fftConfig["bpfoHighOrder"],
        fftConfig["orderToleranceFraction"],
    )
    bpfiBand = orderBand(
        fundamentalHz,
        fftConfig["bpfiLowOrder"],
        fftConfig["bpfiHighOrder"],
        fftConfig["orderToleranceFraction"],
    )
    combinedSignals = {
        "frequencyAxis": frequencyAxis,
        "spectrum": combinedSpectrum,
        "fundamentalHz": fundamentalHz,
        "fundamentalMag": fundamentalMag,
        "bpfoBand": bpfoBand,
        "bpfiBand": bpfiBand,
        "bpfoBandRms": bandRms(
            frequencyAxis,
            combinedSpectrum,
            bpfoBand[0],
            bpfoBand[1],
        ),
        "bpfiBandRms": bandRms(
            frequencyAxis,
            combinedSpectrum,
            bpfiBand[0],
            bpfiBand[1],
        ),
    }
    return combinedSignals


# Stores one set of X, Y, and Z FFT summaries in the frequency dict.
def storeAxisFftSigs(freqSignals, signalPrefix, axisSignals):
    for axisName, axisSignalData in axisSignals.items():
        freqSignals[f"{signalPrefix}{axisName}Spectrum"] = (
            axisSignalData["spectrum"]
        )
        freqSignals[f"{signalPrefix}{axisName}FundamentalHz"] = (
            axisSignalData["fundamentalHz"]
        )
        freqSignals[f"{signalPrefix}{axisName}FundamentalMag"] = (
            axisSignalData["fundamentalMag"]
        )
    return freqSignals


# Builds the single DS18B20 average and gradient signals.
def tempSigs(rawSignals, sampleRate, tempWindowSamples):
    sample = rawSignals["sample"]
    averageSignal = movAvg(rawSignals["ds18b20"], tempWindowSamples)
    averageSample = trimSamp(sample, averageSignal.size)
    gradient = gradSig(averageSignal, sampleRate)
    timeSignals = {
        "ds18b20Avg": averageSignal,
        "ds18b20AvgSample": averageSample,
        "ds18b20Grad": gradient,
        "ds18b20GradSample": averageSample,
    }
    return timeSignals


# Builds magnitude-based time-domain signals for both MPUs.
def magSigs(rawSignals):
    timeSignals = {
        "sample": rawSignals["sample"],
        "mpu1AccMag": mag3(
            rawSignals["mpu1AccX"],
            rawSignals["mpu1AccY"],
            rawSignals["mpu1AccZ"],
        ),
        "mpu1GyrMag": mag3(
            rawSignals["mpu1GyrX"],
            rawSignals["mpu1GyrY"],
            rawSignals["mpu1GyrZ"],
        ),
        "mpu2AccMag": mag3(
            rawSignals["mpu2AccX"],
            rawSignals["mpu2AccY"],
            rawSignals["mpu2AccZ"],
        ),
        "mpu2GyrMag": mag3(
            rawSignals["mpu2GyrX"],
            rawSignals["mpu2GyrY"],
            rawSignals["mpu2GyrZ"],
        ),
    }
    return timeSignals


# Builds acceleration FFT summaries for both MPUs.
def freqSigs(rawSignals, sampleRate, fftConfig):
    freqSignals = {
        "frequencyAxis": None,
    }
    sensorAxes = [
        ("mpu1Acc", "mpu1AccX", "mpu1AccY", "mpu1AccZ"),
        ("mpu2Acc", "mpu2AccX", "mpu2AccY", "mpu2AccZ"),
        ("mpu1Gyr", "mpu1GyrX", "mpu1GyrY", "mpu1GyrZ"),
        ("mpu2Gyr", "mpu2GyrX", "mpu2GyrY", "mpu2GyrZ"),
    ]
    for signalPrefix, axisXKey, axisYKey, axisZKey in sensorAxes:
        axisSignals = axisFftSigs(
            rawSignals[axisXKey],
            rawSignals[axisYKey],
            rawSignals[axisZKey],
            sampleRate,
            fftConfig,
        )
        if freqSignals["frequencyAxis"] is None:
            freqSignals["frequencyAxis"] = axisSignals["X"]["frequencyAxis"]
        freqSignals = storeAxisFftSigs(
            freqSignals,
            signalPrefix,
            axisSignals,
        )
        if "Acc" in signalPrefix:
            combinedSignals = combAxisFftSigs(axisSignals, fftConfig)
            freqSignals[f"{signalPrefix}Spectrum"] = (
                combinedSignals["spectrum"]
            )
            freqSignals[f"{signalPrefix}FundamentalHz"] = (
                combinedSignals["fundamentalHz"]
            )
            freqSignals[f"{signalPrefix}FundamentalMag"] = (
                combinedSignals["fundamentalMag"]
            )
            freqSignals[f"{signalPrefix}BpfoBand"] = (
                combinedSignals["bpfoBand"]
            )
            freqSignals[f"{signalPrefix}BpfiBand"] = (
                combinedSignals["bpfiBand"]
            )
            freqSignals[f"{signalPrefix}BpfoBandRms"] = (
                combinedSignals["bpfoBandRms"]
            )
            freqSignals[f"{signalPrefix}BpfiBandRms"] = (
                combinedSignals["bpfiBandRms"]
            )
    return freqSignals


# Builds metadata for the current signal pass.
def sigMeta(
    bufferData,
    sampleRate,
    periodSeconds,
    tempWindowSeconds,
    fftConfig,
):
    tempWindowSamples = int(sampleRate * tempWindowSeconds)
    signalMeta = {
        "sampleRate": sampleRate,
        "bufferSeconds": bufferData["bufferMeta"]["bufferSeconds"],
        "sampleCount": bufferData["bufferMeta"]["sampleCount"],
        "periodSeconds": periodSeconds,
        "periodSampleCount": int(sampleRate * periodSeconds),
        "tempWindowSeconds": tempWindowSeconds,
        "tempWindowSamples": tempWindowSamples,
        "fftConfig": fftConfig,
    }
    return signalMeta


# Builds named raw, time, and frequency signals from the current buffer.
def buildSigs(
    bufferData,
    sampleRate,
    periodSeconds,
    tempWindowSeconds,
    fftConfig,
):
    signalMeta = sigMeta(
        bufferData,
        sampleRate,
        periodSeconds,
        tempWindowSeconds,
        fftConfig,
    )
    periodBufferData = periodBuf(
        bufferData,
        sampleRate,
        periodSeconds,
    )
    rawSignals = rawSigs(periodBufferData)
    magnitudeSignals = magSigs(rawSignals)
    temperatureSignals = tempSigs(
        rawSignals,
        sampleRate,
        signalMeta["tempWindowSamples"],
    )
    timeSignals = {}
    timeSignals.update(magnitudeSignals)
    timeSignals.update(temperatureSignals)
    freqSignals = freqSigs(
        rawSignals,
        sampleRate,
        fftConfig,
    )
    signalData = {
        "bufferFrame": bufferData["bufferFrame"],
        "rawSignals": rawSignals,
        "timeSignals": timeSignals,
        "freqSignals": freqSignals,
        "signalMeta": signalMeta,
        "bufferMeta": bufferData["bufferMeta"],
        "streamDiagnostics": bufferData.get("streamDiagnostics"),
    }
    return signalData


# Reads a static CSV file into a DataFrame, then builds named signals.
def readCSV(
    csvPath,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    startRow,
    periodSeconds,
    tempWindowSeconds,
    fftConfig,
):
    dataFrame = pd.read_csv(csvPath)
    bufferData = buffer.buildBuf(
        dataFrame,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        startRow,
    )
    return buildSigs(
        bufferData,
        sampleRate,
        periodSeconds,
        tempWindowSeconds,
        fftConfig,
    )


# Reads live sensor samples from the MCU and builds named signals.
def liveSense(
    port,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    baudrate,
    timeout,
    batchSeconds,
    batchHeader,
    packetFormat,
    packetSize,
    maxIdleSeconds,
    initialHeaderIdleSeconds,
    maxConsecutiveBadBatches,
    periodSeconds,
    tempWindowSeconds,
    fftConfig,
):
    bufferData = buffer.liveBuf(
        port,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        baudrate,
        timeout,
        batchSeconds,
        batchHeader,
        packetFormat,
        packetSize,
        maxIdleSeconds,
        initialHeaderIdleSeconds,
        maxConsecutiveBadBatches,
    )
    return buildSigs(
        bufferData,
        sampleRate,
        periodSeconds,
        tempWindowSeconds,
        fftConfig,
    )
