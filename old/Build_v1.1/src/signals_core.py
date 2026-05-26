################################################################################
# Imports                                                                      #
################################################################################
import numpy as np
from scipy import fft

################################################################################
# variables/constants                                                          #
################################################################################
signalCols = [
    "t_us",
    "t_s",
    "ax1",
    "ay1",
    "az1",
    "gx1",
    "gy1",
    "gz1",
    "ax2",
    "ay2",
    "az2",
    "gx2",
    "gy2",
    "gz2",
    "tempC",
]
fftConfig = {
    "minHz": 1.0,
    "maxHz": 60.0,
    "bpfoLowOrder": 3.0,
    "bpfoHighOrder": 5.0,
    "bpfiLowOrder": 5.0,
    "bpfiHighOrder": 7.0,
    "tolFraction": 0.10,
}

################################################################################
# helpers                                                                      #
################################################################################


# Converts the selected dataframe columns into NumPy float64 arrays.
def rawArrays(dataFrame):
    rawSignals = {}
    for signalCol in signalCols:
        rawSignal = dataFrame[signalCol].to_numpy(dtype=np.float64)
        if signalCol == "tempC":
            rawSignal = np.nan_to_num(rawSignal, nan=0.0)
        rawSignals[signalCol] = rawSignal
    return rawSignals


# Calculates the magnitude of one 3-axis signal.
def mag3(axisX, axisY, axisZ):
    magnitude = np.sqrt(axisX**2 + axisY**2 + axisZ**2)
    return magnitude


# Calculates one simple moving average on a 1D signal.
def movAvg(signal, windowSize):
    kernel = np.ones(windowSize, dtype=np.float64) / windowSize
    averagedSignal = np.convolve(signal, kernel, mode="valid")
    return averagedSignal


# Trims one time axis to match a reduced signal length.
def trimTime(timeAxis, reducedSize):
    reducedTime = timeAxis[-reducedSize:]
    return reducedTime


# Calculates one first derivative against sample rate.
def gradSig(signal, sampleRate):
    gradient = np.gradient(signal) * sampleRate
    return gradient


# Builds one DC-removed FFT magnitude spectrum.
def buildSpectrum(signal, sampleRate):
    sampleCount = signal.size
    centeredSignal = signal - np.mean(signal)
    spectrum = fft.rfft(centeredSignal)
    magnitude = (2.0 / sampleCount) * np.abs(spectrum)
    freqAxis = fft.rfftfreq(sampleCount, d=1.0 / sampleRate)
    return freqAxis, magnitude


# Finds one dominant peak inside the selected frequency window.
def fundamentalPeak(freqAxis, magnitude, minHz, maxHz):
    fundMask = (freqAxis >= minHz) & (freqAxis <= maxHz)
    fundFreq = freqAxis[fundMask]
    fundMag = magnitude[fundMask]
    peakIndex = int(np.argmax(fundMag))
    peakHz = float(fundFreq[peakIndex])
    peakMag = float(fundMag[peakIndex])
    return peakHz, peakMag


# Builds one RMS magnitude for one selected band.
def bandRms(freqAxis, magnitude, lowHz, highHz):
    bandMask = (freqAxis >= lowHz) & (freqAxis <= highHz)
    bandMag = magnitude[bandMask]
    rmsValue = float(np.sqrt(np.mean(bandMag**2)))
    return rmsValue


# Builds one order band with a symmetric tolerance.
def orderBand(fundHz, lowOrder, highOrder, tolFraction):
    lowHz = (fundHz * lowOrder) * (1.0 - tolFraction)
    highHz = (fundHz * highOrder) * (1.0 + tolFraction)
    return lowHz, highHz


# Builds one clean time axis from the fixed sample rate.
def timeAxis(rawSignals, sampleRateValue):
    sampleCount = rawSignals["t_us"].size
    timeValue = np.arange(sampleCount, dtype=np.float64) / sampleRateValue
    return timeValue


################################################################################
# main functions                                                               #
################################################################################


# Builds the shared time-domain signal dictionary.
def timeData(rawSignals, sampleRateValue, tempWindowSeconds):
    tempWindowSize = int(sampleRateValue * tempWindowSeconds)
    timeValue = timeAxis(rawSignals, sampleRateValue)
    tempAvg = movAvg(rawSignals["tempC"], tempWindowSize)
    tempTime = trimTime(timeValue, tempAvg.size)
    tempGrad = gradSig(tempAvg, sampleRateValue)
    timeSignals = {
        "t_s": timeValue,
        "acc1Mag": mag3(
            rawSignals["ax1"],
            rawSignals["ay1"],
            rawSignals["az1"],
        ),
        "gyr1Mag": mag3(
            rawSignals["gx1"],
            rawSignals["gy1"],
            rawSignals["gz1"],
        ),
        "acc2Mag": mag3(
            rawSignals["ax2"],
            rawSignals["ay2"],
            rawSignals["az2"],
        ),
        "gyr2Mag": mag3(
            rawSignals["gx2"],
            rawSignals["gy2"],
            rawSignals["gz2"],
        ),
        "tempAvg": tempAvg,
        "tempGrad": tempGrad,
        "tempTime": tempTime,
    }
    return timeSignals


# Builds one X, Y, Z FFT summary group.
def axisData(axisX, axisY, axisZ, sampleRateValue, fftConfig):
    axisSignals = {}
    for axisName, axisSignal in [
        ("X", axisX),
        ("Y", axisY),
        ("Z", axisZ),
    ]:
        freqAxis, spectrum = buildSpectrum(axisSignal, sampleRateValue)
        fundHz, fundMag = fundamentalPeak(
            freqAxis,
            spectrum,
            fftConfig["minHz"],
            fftConfig["maxHz"],
        )
        axisSignals[axisName] = {
            "freqAxis": freqAxis,
            "spectrum": spectrum,
            "fundHz": fundHz,
            "fundMag": fundMag,
        }
    return axisSignals


# Builds one combined acceleration FFT summary from X, Y, and Z.
def combData(axisSignals, fftConfig):
    freqAxis = axisSignals["X"]["freqAxis"]
    spectrum = np.sqrt(
        axisSignals["X"]["spectrum"]**2
        + axisSignals["Y"]["spectrum"]**2
        + axisSignals["Z"]["spectrum"]**2
    )
    fundHz, fundMag = fundamentalPeak(
        freqAxis,
        spectrum,
        fftConfig["minHz"],
        fftConfig["maxHz"],
    )
    bpfoBand = orderBand(
        fundHz,
        fftConfig["bpfoLowOrder"],
        fftConfig["bpfoHighOrder"],
        fftConfig["tolFraction"],
    )
    bpfiBand = orderBand(
        fundHz,
        fftConfig["bpfiLowOrder"],
        fftConfig["bpfiHighOrder"],
        fftConfig["tolFraction"],
    )
    combSignals = {
        "freqAxis": freqAxis,
        "spectrum": spectrum,
        "fundHz": fundHz,
        "fundMag": fundMag,
        "bpfoBand": bpfoBand,
        "bpfiBand": bpfiBand,
        "bpfoRms": bandRms(freqAxis, spectrum, bpfoBand[0], bpfoBand[1]),
        "bpfiRms": bandRms(freqAxis, spectrum, bpfiBand[0], bpfiBand[1]),
    }
    return combSignals


# Builds the shared frequency-domain signal dictionary.
def freqData(rawSignals, sampleRateValue, fftConfig):
    freqSignals = {}
    sensorAxes = [
        ("acc1", "ax1", "ay1", "az1"),
        ("gyr1", "gx1", "gy1", "gz1"),
        ("acc2", "ax2", "ay2", "az2"),
        ("gyr2", "gx2", "gy2", "gz2"),
    ]
    for signalName, axisXKey, axisYKey, axisZKey in sensorAxes:
        axisSignals = axisData(
            rawSignals[axisXKey],
            rawSignals[axisYKey],
            rawSignals[axisZKey],
            sampleRateValue,
            fftConfig,
        )
        if "freqAxis" not in freqSignals:
            freqSignals["freqAxis"] = axisSignals["X"]["freqAxis"]
        for axisName, axisSignal in axisSignals.items():
            freqSignals[f"{signalName}{axisName}Spectrum"] = (
                axisSignal["spectrum"]
            )
            freqSignals[f"{signalName}{axisName}FundHz"] = (
                axisSignal["fundHz"]
            )
            freqSignals[f"{signalName}{axisName}FundMag"] = (
                axisSignal["fundMag"]
            )
        if "acc" in signalName:
            combSignals = combData(axisSignals, fftConfig)
            freqSignals[f"{signalName}Spectrum"] = combSignals["spectrum"]
            freqSignals[f"{signalName}FundHz"] = combSignals["fundHz"]
            freqSignals[f"{signalName}FundMag"] = combSignals["fundMag"]
            freqSignals[f"{signalName}BpfoBand"] = combSignals["bpfoBand"]
            freqSignals[f"{signalName}BpfiBand"] = combSignals["bpfiBand"]
            freqSignals[f"{signalName}BpfoRms"] = combSignals["bpfoRms"]
            freqSignals[f"{signalName}BpfiRms"] = combSignals["bpfiRms"]
    return freqSignals
