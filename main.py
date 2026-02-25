from pathlib import Path

import charting
import pandas as pd
import signals
import sys

csvPath = "data/testData/nivel2.csv"
outputDirPath = "outputs/csv"
serialPort = "COM3"
baudRate = 115200
serialDelimiter = ","
serialTimeout = 1.0
serialEncoding = "utf-8"


# Orchestrates CSV input through signals, DSP, then analytics.
def runCSV(csv_path):
    csvData = pd.read_csv(csv_path)
    print("csvPath:", csv_path)
    print("csvDataHead:")
    print(csvData.head())

    (
        sample,
        Acel_X,
        Acel_Y,
        Acel_Z,
        Giro_X,
        Giro_Y,
        Giro_Z,
        Temperatura,
    ) = signals.buildSignals(csvData)
    print("sample[:5]:", sample[:5])
    print("Acel_X[:5]:", Acel_X[:5])
    print("Acel_Y[:5]:", Acel_Y[:5])
    print("Acel_Z[:5]:", Acel_Z[:5])
    print("Giro_X[:5]:", Giro_X[:5])
    print("Giro_Y[:5]:", Giro_Y[:5])
    print("Giro_Z[:5]:", Giro_Z[:5])
    print("Temperatura[:5]:", Temperatura[:5])
    plotRawPath = str(
        Path(outputDirPath) / (Path(csv_path).stem + "_plotRaw.png")
    )
    charting.plotRaw(
        sample,
        Acel_X,
        Acel_Y,
        Acel_Z,
        Giro_X,
        Giro_Y,
        Giro_Z,
        Temperatura,
        plotRawPath,
    )
    print("plotRawPath:", plotRawPath)

    # DSP and analytics are disabled for this signal-chain test.
    return (
        sample,
        Acel_X,
        Acel_Y,
        Acel_Z,
        Giro_X,
        Giro_Y,
        Giro_Z,
        Temperatura,
    )


# Orchestrates live serial input through signals, DSP, then analytics.
def runLive(
    port,
    baudrate,
    delimiter,
    timeout,
    encoding,
):
    (
        sample,
        Acel_X,
        Acel_Y,
        Acel_Z,
        Giro_X,
        Giro_Y,
        Giro_Z,
        Temperatura,
    ) = signals.liveSense(
        port=port,
        sample_count=5000,
        baudrate=baudrate,
        delimiter=delimiter,
        timeout=timeout,
        encoding=encoding,
    )
    print("sample[:5]:", sample[:5])
    print("Acel_X[:5]:", Acel_X[:5])
    print("Acel_Y[:5]:", Acel_Y[:5])
    print("Acel_Z[:5]:", Acel_Z[:5])
    print("Giro_X[:5]:", Giro_X[:5])
    print("Giro_Y[:5]:", Giro_Y[:5])
    print("Giro_Z[:5]:", Giro_Z[:5])
    print("Temperatura[:5]:", Temperatura[:5])

    # DSP and analytics are disabled for this signal-chain test.
    return (
        sample,
        Acel_X,
        Acel_Y,
        Acel_Z,
        Giro_X,
        Giro_Y,
        Giro_Z,
        Temperatura,
    )


if __name__ == "__main__":
    source = sys.argv[1]

    if source == "csv":
        pipelineOutput = runCSV(csvPath)
    else:
        pipelineOutput = runLive(
            port=serialPort,
            baudrate=baudRate,
            delimiter=serialDelimiter,
            timeout=serialTimeout,
            encoding=serialEncoding,
        )

    print("pipelineOutputReady")
