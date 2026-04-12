################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import sys

import pandas as pd

import charting_core
import signals_core

################################################################################
# variables/constants                                                          #
################################################################################
csvPath = "ML_data/test1.csv"
outDir = "outputs/simpleCharts"
sampleRate = 1000.0

################################################################################
# helpers                                                                      #
################################################################################


# Creates the chart output directories used by analysis.
def chartDirs(outDir):
    outputDirs = [
        Path(outDir),
        Path(outDir) / "sensor",
        Path(outDir) / "allFFT",
    ]
    for outputDir in outputDirs:
        outputDir.mkdir(parents=True, exist_ok=True)


################################################################################
# main functions                                                               #
################################################################################


# Runs pure CSV signal analysis and saves the chart outputs.
def main():
    csvFile = csvPath
    if len(sys.argv) > 1:
        csvFile = sys.argv[1]

    dataFrame = pd.read_csv(csvFile)
    rawSignals = signals_core.rawArrays(dataFrame)
    timeSignals = signals_core.timeData(rawSignals, sampleRate)
    freqSignals = signals_core.freqData(
        rawSignals,
        sampleRate,
        signals_core.fftConfig,
    )

    chartDirs(outDir)
    sensorPath = str(Path(outDir) / "sensor" / "sensor.png")
    fftGridPath = str(Path(outDir) / "allFFT" / "fftGrid.png")

    charting_core.plotSensor(rawSignals, timeSignals, freqSignals, sensorPath)
    charting_core.plotAxisGrid(freqSignals, fftGridPath)

    print("csvPath:", csvFile)
    print("sampleRate:", sampleRate)
    print("sensorPath:", sensorPath)
    print("fftGridPath:", fftGridPath)


if __name__ == "__main__":
    main()
