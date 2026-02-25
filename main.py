from pathlib import Path

import analytics
import charting
import pandas as pd
import signals
import sys

csvPath = "data/testData/nivel2.csv"
outputDirPath = "outputs/csv"
tempTrendNth = 10
serialPort = "COM3"
baudRate = 115200
serialDelimiter = ","
serialTimeout = 1.0
serialEncoding = "utf-8"


# Orchestrates CSV input through signals, analytics, then charting.
def runCSV(csv_path):
    csvData = pd.read_csv(csv_path)
    print("csvPath:", csv_path)
    print("csvDataHead:")
    print(csvData.head())

    rawSignals = signals.buildSignals(csvData)
    dspSignals = analytics.buildDSPSignals(rawSignals, tempTrendNth)

    sample = rawSignals[0]
    Acel_X = rawSignals[1]
    Acel_Y = rawSignals[2]
    Acel_Z = rawSignals[3]
    Giro_X = rawSignals[4]
    Giro_Y = rawSignals[5]
    Giro_Z = rawSignals[6]
    Temperatura = rawSignals[7]
    tempTrend = dspSignals[9]

    print("sample[:5]:", sample[:5])
    print("Acel_X[:5]:", Acel_X[:5])
    print("Acel_Y[:5]:", Acel_Y[:5])
    print("Acel_Z[:5]:", Acel_Z[:5])
    print("Giro_X[:5]:", Giro_X[:5])
    print("Giro_Y[:5]:", Giro_Y[:5])
    print("Giro_Z[:5]:", Giro_Z[:5])
    print("Temperatura[:5]:", Temperatura[:5])
    print("tempTrendNth:", tempTrendNth)
    print("tempTrend[:5]:", tempTrend[:5])

    plotRawPath = str(
        Path(outputDirPath) / (Path(csv_path).stem + "_plotRaw.png")
    )
    charting.plotRaw(dspSignals, plotRawPath)
    print("plotRawPath:", plotRawPath)

    return dspSignals


# Orchestrates live input through signals and analytics.
def runLive(
    port,
    baudrate,
    delimiter,
    timeout,
    encoding,
):
    rawSignals = signals.liveSense(
        port=port,
        sample_count=5000,
        baudrate=baudrate,
        delimiter=delimiter,
        timeout=timeout,
        encoding=encoding,
    )
    dspSignals = analytics.buildDSPSignals(rawSignals, tempTrendNth)

    sample = rawSignals[0]
    Acel_X = rawSignals[1]
    Acel_Y = rawSignals[2]
    Acel_Z = rawSignals[3]
    Giro_X = rawSignals[4]
    Giro_Y = rawSignals[5]
    Giro_Z = rawSignals[6]
    Temperatura = rawSignals[7]
    tempTrend = dspSignals[9]

    print("sample[:5]:", sample[:5])
    print("Acel_X[:5]:", Acel_X[:5])
    print("Acel_Y[:5]:", Acel_Y[:5])
    print("Acel_Z[:5]:", Acel_Z[:5])
    print("Giro_X[:5]:", Giro_X[:5])
    print("Giro_Y[:5]:", Giro_Y[:5])
    print("Giro_Z[:5]:", Giro_Z[:5])
    print("Temperatura[:5]:", Temperatura[:5])
    print("tempTrendNth:", tempTrendNth)
    print("tempTrend[:5]:", tempTrend[:5])

    return dspSignals


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
