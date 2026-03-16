import numpy as np
import pandas as pd
import serial

import buffer


# Converts named sensor columns into NumPy float64 arrays.
def buildRawSignals(bufferData):
    rawSignals = {}
    for sensorColumn, sensorArray in bufferData["bufferArrays"].items():
        rawSignals[sensorColumn] = sensorArray.astype(np.float64)
    return rawSignals


# Calculates the magnitude of a 3-axis signal.
def buildMagnitude(axisX, axisY, axisZ):
    magnitude = np.sqrt(axisX**2 + axisY**2 + axisZ**2)
    return magnitude


# Calculates a simple moving average on a 1D array.
def movingAverage(signal, windowSize):
    kernel = np.ones(windowSize, dtype=np.float64) / windowSize
    averagedSignal = np.convolve(signal, kernel, mode="valid")
    return averagedSignal


# Builds a matching sample axis for a reduced-length signal.
def buildReducedSample(sample, reducedSize):
    reducedSample = sample[-reducedSize:]
    return reducedSample


# Calculates the first derivative of a 1D array against time.
def gradientSignal(signal, sampleRate):
    gradient = np.gradient(signal) * sampleRate
    return gradient


# Calculates the real FFT magnitude of a 1D array.
def buildSpectrum(signal):
    centeredSignal = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centeredSignal))
    return spectrum


# Builds one summed axis-power spectrum from three raw axes.
def buildAxisPowerSpectrum(axisX, axisY, axisZ):
    axisXSpectrum = buildSpectrum(axisX)
    axisYSpectrum = buildSpectrum(axisY)
    axisZSpectrum = buildSpectrum(axisZ)
    axisPowerSpectrum = (
        axisXSpectrum**2
        + axisYSpectrum**2
        + axisZSpectrum**2
    )
    return axisPowerSpectrum


# Builds the frequency axis for a real FFT.
def buildFrequencyAxis(sampleCount, sampleRate):
    frequencyAxis = np.fft.rfftfreq(sampleCount, d=1.0 / sampleRate)
    return frequencyAxis


# Finds the dominant frequency in a magnitude spectrum.
def buildDominantFrequency(frequencyAxis, spectrum):
    dominantIndex = np.argmax(spectrum[1:]) + 1
    dominantFrequency = frequencyAxis[dominantIndex]
    return dominantFrequency


# Finds the dominant magnitude in a magnitude spectrum.
def buildDominantMagnitude(spectrum):
    dominantMagnitude = np.max(spectrum[1:])
    return dominantMagnitude


# Calculates the energy inside one frequency band.
def buildBandEnergy(frequencyAxis, spectrum, bandLow, bandHigh):
    bandMask = (frequencyAxis >= bandLow) & (frequencyAxis <= bandHigh)
    bandEnergy = np.sum(spectrum[bandMask] ** 2)
    return bandEnergy


# Calculates named band energies for one spectrum.
def buildBandEnergies(frequencyAxis, spectrum, frequencyBands):
    bandEnergies = {}
    for bandName, bandLow, bandHigh in frequencyBands:
        bandEnergy = buildBandEnergy(
            frequencyAxis,
            spectrum,
            bandLow,
            bandHigh,
        )
        bandEnergies[bandName] = bandEnergy
    return bandEnergies


# Builds temperature average and gradient signals.
def buildTemperatureSignals(rawSignals, sampleRate, tempWindowSamples):
    sample = rawSignals["sample"]
    timeSignals = {}
    for tempKey in [
        "mpu1Temp",
        "mpu2Temp",
        "ds18b20One",
        "ds18b20Two",
    ]:
        averageKey = f"{tempKey}Avg"
        gradientKey = f"{tempKey}Grad"
        averageSignal = movingAverage(rawSignals[tempKey], tempWindowSamples)
        averageSample = buildReducedSample(sample, averageSignal.size)
        gradient = gradientSignal(averageSignal, sampleRate)
        timeSignals[averageKey] = averageSignal
        timeSignals[f"{averageKey}Sample"] = averageSample
        timeSignals[gradientKey] = gradient
        timeSignals[f"{gradientKey}Sample"] = averageSample
    return timeSignals


# Builds magnitude-based time-domain signals for both MPUs.
def buildMagnitudeSignals(rawSignals):
    timeSignals = {
        "sample": rawSignals["sample"],
        "mpu1AccMag": buildMagnitude(
            rawSignals["mpu1AccX"],
            rawSignals["mpu1AccY"],
            rawSignals["mpu1AccZ"],
        ),
        "mpu1GyrMag": buildMagnitude(
            rawSignals["mpu1GyrX"],
            rawSignals["mpu1GyrY"],
            rawSignals["mpu1GyrZ"],
        ),
        "mpu2AccMag": buildMagnitude(
            rawSignals["mpu2AccX"],
            rawSignals["mpu2AccY"],
            rawSignals["mpu2AccZ"],
        ),
        "mpu2GyrMag": buildMagnitude(
            rawSignals["mpu2GyrX"],
            rawSignals["mpu2GyrY"],
            rawSignals["mpu2GyrZ"],
        ),
    }
    return timeSignals


# Builds frequency-domain signals from magnitude and raw axis signals.
def buildFrequencySignals(rawSignals, timeSignals, sampleRate, frequencyBands):
    sampleCount = timeSignals["sample"].size
    frequencyAxis = buildFrequencyAxis(sampleCount, sampleRate)
    freqSignals = {
        "frequencyAxis": frequencyAxis,
    }
    signalGroups = [
        ("mpu1Acc", "mpu1AccMag", "mpu1AccX", "mpu1AccY", "mpu1AccZ"),
        ("mpu1Gyr", "mpu1GyrMag", "mpu1GyrX", "mpu1GyrY", "mpu1GyrZ"),
        ("mpu2Acc", "mpu2AccMag", "mpu2AccX", "mpu2AccY", "mpu2AccZ"),
        ("mpu2Gyr", "mpu2GyrMag", "mpu2GyrX", "mpu2GyrY", "mpu2GyrZ"),
    ]
    for signalPrefix, magnitudeKey, axisXKey, axisYKey, axisZKey in (
        signalGroups
    ):
        magnitudeSpectrum = buildSpectrum(timeSignals[magnitudeKey])
        dominantFrequency = buildDominantFrequency(
            frequencyAxis,
            magnitudeSpectrum,
        )
        dominantMagnitude = buildDominantMagnitude(magnitudeSpectrum)
        axisPowerSpectrum = buildAxisPowerSpectrum(
            rawSignals[axisXKey],
            rawSignals[axisYKey],
            rawSignals[axisZKey],
        )
        bandEnergies = buildBandEnergies(
            frequencyAxis,
            axisPowerSpectrum,
            frequencyBands,
        )
        freqSignals[f"{magnitudeKey}Spectrum"] = magnitudeSpectrum
        freqSignals[f"{magnitudeKey}DomFreq"] = dominantFrequency
        freqSignals[f"{magnitudeKey}DomMag"] = dominantMagnitude
        freqSignals[f"{signalPrefix}AxisPowerSpectrum"] = axisPowerSpectrum
        for bandName, bandEnergy in bandEnergies.items():
            freqSignals[f"{signalPrefix}AxisPower{bandName}"] = bandEnergy
    return freqSignals


# Builds metadata for the current signal pass.
def buildSignalMeta(bufferData, sampleRate, tempWindowSeconds, frequencyBands):
    tempWindowSamples = int(sampleRate * tempWindowSeconds)
    signalMeta = {
        "sampleRate": sampleRate,
        "bufferSeconds": bufferData["bufferMeta"]["bufferSeconds"],
        "sampleCount": bufferData["bufferMeta"]["sampleCount"],
        "tempWindowSeconds": tempWindowSeconds,
        "tempWindowSamples": tempWindowSamples,
        "frequencyBands": frequencyBands,
    }
    return signalMeta


# Builds named raw, time, and frequency signals from the current buffer.
def buildSignals(bufferData, sampleRate, tempWindowSeconds, frequencyBands):
    signalMeta = buildSignalMeta(
        bufferData,
        sampleRate,
        tempWindowSeconds,
        frequencyBands,
    )
    rawSignals = buildRawSignals(bufferData)
    magnitudeSignals = buildMagnitudeSignals(rawSignals)
    temperatureSignals = buildTemperatureSignals(
        rawSignals,
        sampleRate,
        signalMeta["tempWindowSamples"],
    )
    timeSignals = {}
    timeSignals.update(magnitudeSignals)
    timeSignals.update(temperatureSignals)
    freqSignals = buildFrequencySignals(
        rawSignals,
        magnitudeSignals,
        sampleRate,
        frequencyBands,
    )
    signalData = {
        "bufferFrame": bufferData["bufferFrame"],
        "rawSignals": rawSignals,
        "timeSignals": timeSignals,
        "freqSignals": freqSignals,
        "signalMeta": signalMeta,
        "bufferMeta": bufferData["bufferMeta"],
    }
    return signalData


# Reads a static CSV file into a DataFrame, then builds named signals.
def readCSV(
    csvPath,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    startRow,
    tempWindowSeconds,
    frequencyBands,
):
    dataFrame = pd.read_csv(csvPath)
    bufferData = buffer.buildBuffer(
        dataFrame,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        startRow,
    )
    return buildSignals(
        bufferData,
        sampleRate,
        tempWindowSeconds,
        frequencyBands,
    )


# Reads live sensor samples from the MCU and builds named signals.
def liveSense(
    port,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    baudrate,
    delimiter,
    timeout,
    encoding,
    tempWindowSeconds,
    frequencyBands,
):
    sampleCount = int(sampleRate * bufferSeconds)
    rows = []
    with serial.Serial(port=port, baudrate=baudrate, timeout=timeout) as link:
        for _ in range(sampleCount):
            line = link.readline().decode(encoding).strip()
            rows.append(line.split(delimiter))
    dataFrame = pd.DataFrame(rows, columns=sensorColumns)
    bufferData = buffer.buildBuffer(
        dataFrame,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        0,
    )
    return buildSignals(
        bufferData,
        sampleRate,
        tempWindowSeconds,
        frequencyBands,
    )


# Returns the current motor signal column order.
def buildMotorColumns():
    return buffer.buildMotorColumns()
