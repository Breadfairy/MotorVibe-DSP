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
csvPath = "data/motordata/tilt10mm.csv"
outDir = "outputs/simpleCharts"
sampleRate = 1000.0
tempWinSecs = 0.01

################################################################################
# helpers                                                                      #
################################################################################


# Creates the chart output directories used by analysis.
def chartDirs(outDir):
    outputDirs = [
        Path(outDir),
        Path(outDir) / "sensors1",
        Path(outDir) / "sensors2",
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
    timeSignals = signals_core.timeData(
        rawSignals,
        sampleRate,
        tempWinSecs,
    )
    freqSignals = signals_core.freqData(
        rawSignals,
        sampleRate,
        signals_core.fftConfig,
    )

    chartDirs(outDir)
    sensor1Path = str(Path(outDir) / "sensors1" / "sensor1.png")
    sensor2Path = str(Path(outDir) / "sensors2" / "sensor2.png")
    fftGridPath = str(Path(outDir) / "allFFT" / "fftGrid.png")

    charting_core.plotSensor(
        rawSignals,
        timeSignals,
        freqSignals,
        1,
        sensor1Path,
    )
    charting_core.plotSensor(
        rawSignals,
        timeSignals,
        freqSignals,
        2,
        sensor2Path,
    )
    charting_core.plotAxisGrid(freqSignals, fftGridPath)

    print("csvPath:", csvFile)
    print("sampleRate:", sampleRate)
    print("sensor1Path:", sensor1Path)
    print("sensor2Path:", sensor2Path)
    print("fftGridPath:", fftGridPath)


if __name__ == "__main__":
    main()
