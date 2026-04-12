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
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
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
        rawSignals[signalCol] = dataFrame[signalCol].to_numpy(
            dtype=np.float64
        )
    return rawSignals


# Calculates the magnitude of one 3-axis signal.
def mag3(axisX, axisY, axisZ):
    magnitude = np.sqrt(axisX**2 + axisY**2 + axisZ**2)
    return magnitude


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
def timeData(rawSignals, sampleRateValue):
    timeValue = timeAxis(rawSignals, sampleRateValue)
    timeSignals = {
        "t_s": timeValue,
        "accMag": mag3(
            rawSignals["ax"],
            rawSignals["ay"],
            rawSignals["az"],
        ),
        "gyrMag": mag3(
            rawSignals["gx"],
            rawSignals["gy"],
            rawSignals["gz"],
        ),
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
        ("acc", "ax", "ay", "az"),
        ("gyr", "gx", "gy", "gz"),
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
        if signalName == "acc":
            combSignals = combData(axisSignals, fftConfig)
            freqSignals[f"{signalName}Spectrum"] = combSignals["spectrum"]
            freqSignals[f"{signalName}FundHz"] = combSignals["fundHz"]
            freqSignals[f"{signalName}FundMag"] = combSignals["fundMag"]
            freqSignals[f"{signalName}BpfoBand"] = combSignals["bpfoBand"]
            freqSignals[f"{signalName}BpfiBand"] = combSignals["bpfiBand"]
            freqSignals[f"{signalName}BpfoRms"] = combSignals["bpfoRms"]
            freqSignals[f"{signalName}BpfiRms"] = combSignals["bpfiRms"]
    return freqSignals
