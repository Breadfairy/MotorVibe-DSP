import numpy as np
import pandas as pd
import serial


# Returns the current target motor sensor column order.
def buildMotorColumns():
    sensorColumns = [
        "sample",
        "mpu1AccX",
        "mpu1AccY",
        "mpu1AccZ",
        "mpu1GyrX",
        "mpu1GyrY",
        "mpu1GyrZ",
        "mpu1Temp",
        "mpu2AccX",
        "mpu2AccY",
        "mpu2AccZ",
        "mpu2GyrX",
        "mpu2GyrY",
        "mpu2GyrZ",
        "mpu2Temp",
        "ds18b20One",
        "ds18b20Two",
    ]
    return sensorColumns


# Converts the current buffer frame into named NumPy float64 arrays.
def buildBufferArrays(bufferFrame, sensorColumns):
    bufferArrays = {}
    for sensorColumn in sensorColumns:
        bufferArrays[sensorColumn] = bufferFrame[sensorColumn].to_numpy(
            dtype=np.float64
        )
    return bufferArrays


# Builds metadata that describes the current fixed-size buffer window.
def buildBufferMeta(sampleRate, bufferSeconds, startRow, sampleCount):
    bufferMeta = {
        "sampleRate": sampleRate,
        "bufferSeconds": bufferSeconds,
        "sampleCount": sampleCount,
        "startRow": startRow,
        "endRow": startRow + sampleCount,
    }
    return bufferMeta


# Slices a fixed-size buffer window from a DataFrame.
def buildBuffer(dataFrame, sensorColumns, sampleRate, bufferSeconds, startRow):
    sampleCount = int(sampleRate * bufferSeconds)
    endRow = startRow + sampleCount
    bufferFrame = dataFrame.iloc[startRow:endRow].reset_index(drop=True)
    bufferArrays = buildBufferArrays(bufferFrame, sensorColumns)
    bufferMeta = buildBufferMeta(
        sampleRate,
        bufferSeconds,
        startRow,
        sampleCount,
    )
    bufferData = {
        "bufferFrame": bufferFrame,
        "bufferArrays": bufferArrays,
        "bufferMeta": bufferMeta,
    }
    return bufferData


# Converts serial row values into a DataFrame with the active schema.
def buildRowsFrame(rows, sensorColumns):
    rowsFrame = pd.DataFrame(rows, columns=sensorColumns)
    return rowsFrame


# Reads a fixed number of serial rows from an open link.
def readSerialRows(link, rowCount, delimiter, encoding):
    rows = []
    for _ in range(rowCount):
        line = link.readline().decode(encoding).strip()
        rows.append(line.split(delimiter))
    return rows


# Drops the oldest rows and appends new rows onto the current buffer.
def updateRollingBuffer(
    bufferData,
    newRowsFrame,
    sensorColumns,
    sampleRate,
    bufferSeconds,
):
    sampleCount = int(sampleRate * bufferSeconds)
    stepCount = len(newRowsFrame)
    startRow = bufferData["bufferMeta"]["startRow"] + stepCount
    combinedFrame = pd.concat(
        [bufferData["bufferFrame"], newRowsFrame],
        ignore_index=True,
    )
    bufferFrame = combinedFrame.iloc[-sampleCount:].reset_index(drop=True)
    bufferArrays = buildBufferArrays(bufferFrame, sensorColumns)
    bufferMeta = buildBufferMeta(
        sampleRate,
        bufferSeconds,
        startRow,
        sampleCount,
    )
    updatedBufferData = {
        "bufferFrame": bufferFrame,
        "bufferArrays": bufferArrays,
        "bufferMeta": bufferMeta,
    }
    return updatedBufferData


# Reads CSV data and slices the current fixed-size buffer window.
def readCSVBuffer(
    csvPath,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    startRow,
):
    dataFrame = pd.read_csv(csvPath)
    return buildBuffer(
        dataFrame,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        startRow,
    )


# Reads one fixed-size buffer window from a live serial stream.
def liveBuffer(
    port,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    baudrate,
    delimiter,
    timeout,
    encoding,
):
    sampleCount = int(sampleRate * bufferSeconds)
    with serial.Serial(port=port, baudrate=baudrate, timeout=timeout) as link:
        rows = readSerialRows(link, sampleCount, delimiter, encoding)
    dataFrame = buildRowsFrame(rows, sensorColumns)
    return buildBuffer(
        dataFrame,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        0,
    )


# Yields rolling live buffers with a fixed-size window and stepped overlap.
def liveRollingBuffer(
    port,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    stepSeconds,
    baudrate,
    delimiter,
    timeout,
    encoding,
):
    sampleCount = int(sampleRate * bufferSeconds)
    stepCount = int(sampleRate * stepSeconds)
    with serial.Serial(port=port, baudrate=baudrate, timeout=timeout) as link:
        initialRows = readSerialRows(link, sampleCount, delimiter, encoding)
        initialFrame = buildRowsFrame(initialRows, sensorColumns)
        bufferData = buildBuffer(
            initialFrame,
            sensorColumns,
            sampleRate,
            bufferSeconds,
            0,
        )
        yield bufferData

        while True:
            newRows = readSerialRows(link, stepCount, delimiter, encoding)
            newRowsFrame = buildRowsFrame(newRows, sensorColumns)
            bufferData = updateRollingBuffer(
                bufferData,
                newRowsFrame,
                sensorColumns,
                sampleRate,
                bufferSeconds,
            )
            yield bufferData
