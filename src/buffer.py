import struct
import time

import numpy as np
import pandas as pd
import serial


# Returns the current reduced motor sensor column order.
def motorCols():
    sensorColumns = [
        "sample",
        "mpu1AccX",
        "mpu1AccY",
        "mpu1AccZ",
        "mpu1GyrX",
        "mpu1GyrY",
        "mpu1GyrZ",
        "mpu2AccX",
        "mpu2AccY",
        "mpu2AccZ",
        "mpu2GyrX",
        "mpu2GyrY",
        "mpu2GyrZ",
        "ds18b20",
    ]
    return sensorColumns


# Converts the current buffer frame into named NumPy float64 arrays.
def bufArrs(bufferFrame, sensorColumns):
    bufferArrays = {}
    for sensorColumn in sensorColumns:
        bufferArrays[sensorColumn] = bufferFrame[sensorColumn].to_numpy(
            dtype=np.float64
        )
    return bufferArrays


# Builds metadata that describes the current fixed-size buffer window.
def bufMeta(sampleRate, bufferSeconds, startRow, sampleCount):
    bufferMeta = {
        "sampleRate": sampleRate,
        "bufferSeconds": bufferSeconds,
        "sampleCount": sampleCount,
        "startRow": startRow,
        "endRow": startRow + sampleCount,
    }
    return bufferMeta


# Slices a fixed-size buffer window from a DataFrame.
def buildBuf(dataFrame, sensorColumns, sampleRate, bufferSeconds, startRow):
    targetSampleCount = int(sampleRate * bufferSeconds)
    endRow = startRow + targetSampleCount
    bufferFrame = dataFrame.iloc[startRow:endRow][sensorColumns].reset_index(
        drop=True
    )
    sampleCount = bufferFrame.shape[0]
    bufferArrays = bufArrs(bufferFrame, sensorColumns)
    bufferMeta = bufMeta(
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


# Converts row values into a DataFrame with the active schema.
def rowsDf(rows, sensorColumns):
    rowsFrame = pd.DataFrame(rows, columns=sensorColumns)
    return rowsFrame


# Drops the oldest rows and appends new rows onto the current buffer.
def rollBuf(
    bufferData,
    newRowsFrame,
    sensorColumns,
    sampleRate,
    bufferSeconds,
):
    targetSampleCount = int(sampleRate * bufferSeconds)
    stepCount = len(newRowsFrame)
    startRow = bufferData["bufferMeta"]["startRow"] + stepCount
    combinedFrame = pd.concat(
        [bufferData["bufferFrame"], newRowsFrame],
        ignore_index=True,
    )
    bufferFrame = combinedFrame.iloc[-targetSampleCount:].reset_index(
        drop=True
    )
    sampleCount = bufferFrame.shape[0]
    bufferArrays = bufArrs(bufferFrame, sensorColumns)
    bufferMeta = bufMeta(
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
def readCSVBuf(
    csvPath,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    startRow,
):
    dataFrame = pd.read_csv(csvPath)
    return buildBuf(
        dataFrame,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        startRow,
    )


# Builds the serial link with the minimal transport settings.
def openSerialLink(port, baudrate, timeout):
    link = serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=timeout,
    )
    return link


# Reads one exact binary block into the target buffer.
def readBlock(link, blockBuffer, blockByteCount, maxIdleSeconds):
    blockView = memoryview(blockBuffer)
    byteCount = 0
    lastProgressTime = time.perf_counter()
    while byteCount < blockByteCount:
        readCount = link.readinto(blockView[byteCount:blockByteCount])
        if readCount is None:
            continue
        if readCount == 0:
            now = time.perf_counter()
            if (now - lastProgressTime) >= maxIdleSeconds:
                raise TimeoutError("serial stream stalled during block read")
            continue
        byteCount += readCount
        lastProgressTime = time.perf_counter()
    return blockBuffer


# Reads forward until the next fixed batch header is found.
def syncToHeader(link, batchHeader, maxIdleSeconds):
    headerSize = len(batchHeader)
    syncBuffer = bytearray()
    lastProgressTime = time.perf_counter()
    while True:
        nextByte = link.read(1)
        if len(nextByte) == 0:
            now = time.perf_counter()
            if (now - lastProgressTime) >= maxIdleSeconds:
                raise TimeoutError("serial stream stalled while seeking header")
            continue
        syncBuffer += nextByte
        lastProgressTime = time.perf_counter()
        if len(syncBuffer) > headerSize:
            syncBuffer = syncBuffer[-headerSize:]
        if syncBuffer == batchHeader:
            return


# Unpacks one packet from one selected byte offset.
def unpackPacketAt(blockBuffer, byteOffset, packetFormat):
    rowValues = list(struct.unpack_from(packetFormat, blockBuffer, byteOffset))
    return rowValues


# Rebuilds one row so the captured sample starts at 1.
def normalizeRow(rowValues, startSample):
    normalizedRow = list(rowValues)
    normalizedRow[0] = int(rowValues[0] - startSample + 1)
    return normalizedRow


# Builds the current capture sizes from the active sample timing.
def captureSizes(sampleRate, batchSeconds, packetSize, batchHeader):
    batchPacketCount = int(sampleRate * batchSeconds)
    batchPayloadByteCount = batchPacketCount * packetSize
    captureSizes = {
        "batchPacketCount": batchPacketCount,
        "batchHeaderSize": len(batchHeader),
        "batchPayloadByteCount": batchPayloadByteCount,
    }
    return captureSizes


# Reads one fixed framed batch payload after syncing to the batch header.
def readBatchPayload(
    link,
    batchBytes,
    batchHeader,
    batchPayloadByteCount,
    headerIdleSeconds,
    blockIdleSeconds,
):
    syncToHeader(link, batchHeader, headerIdleSeconds)
    readBlock(
        link,
        batchBytes,
        batchPayloadByteCount,
        blockIdleSeconds,
    )
    return batchBytes


# Hard-fails if the packet layout does not match the reduced schema.
def validateBatchRows(batchBytes, batchPacketCount, packetSize, packetFormat):
    previousSample = None
    firstSample = None
    for packetIndex in range(batchPacketCount):
        byteOffset = packetIndex * packetSize
        rowValues = unpackPacketAt(
            batchBytes,
            byteOffset,
            packetFormat,
        )
        sampleValue = rowValues[0]
        if firstSample is None:
            firstSample = sampleValue
        if previousSample is not None and sampleValue != previousSample + 1:
            raise ValueError(
                "live binary packet mismatch. "
                f"packetIndex={packetIndex} "
                f"expectedSample={previousSample + 1} "
                f"actualSample={sampleValue} "
                f"firstSample={firstSample}"
            )
        previousSample = sampleValue


# Decodes one batch payload into normalized row values.
def decodeBatchRows(
    batchBytes,
    batchPacketCount,
    packetSize,
    packetFormat,
    startSample,
):
    validateBatchRows(
        batchBytes,
        batchPacketCount,
        packetSize,
        packetFormat,
    )
    firstRow = unpackPacketAt(batchBytes, 0, packetFormat)
    if startSample is None:
        startSample = firstRow[0]
    rows = []
    for packetIndex in range(batchPacketCount):
        byteOffset = packetIndex * packetSize
        rowValues = unpackPacketAt(
            batchBytes,
            byteOffset,
            packetFormat,
        )
        rows.append(normalizeRow(rowValues, startSample))
    return rows, startSample


# Builds the rolling live stream diagnostics state.
def liveDiagState():
    liveDiagnostics = {
        "goodBatchCount": 0,
        "badBatchCount": 0,
        "consecutiveBadBatchCount": 0,
        "lastBadBatchMessage": None,
    }
    return liveDiagnostics


# Updates the diagnostics state after one good batch.
def markGoodBatch(liveDiagnostics):
    liveDiagnostics["goodBatchCount"] += 1
    liveDiagnostics["consecutiveBadBatchCount"] = 0
    return liveDiagnostics


# Updates the diagnostics state after one bad batch.
def markBadBatch(liveDiagnostics, errorText):
    liveDiagnostics["badBatchCount"] += 1
    liveDiagnostics["consecutiveBadBatchCount"] += 1
    liveDiagnostics["lastBadBatchMessage"] = errorText
    return liveDiagnostics


# Stores the current stream diagnostics on one buffer payload.
def withLiveDiagnostics(bufferData, liveDiagnostics):
    bufferData["streamDiagnostics"] = dict(liveDiagnostics)
    return bufferData


# Reads a fixed number of binary batches into normalized row values.
def readBinaryBatches(
    link,
    batchCount,
    sizes,
    batchHeader,
    packetFormat,
    packetSize,
    maxIdleSeconds,
    initialHeaderIdleSeconds,
    startSample,
    liveDiagnostics,
    maxConsecutiveBadBatches,
):
    batchBytes = bytearray(sizes["batchPayloadByteCount"])
    rows = []
    batchIndex = 0
    batchAttemptCount = 0
    while batchIndex < batchCount:
        headerIdleSeconds = maxIdleSeconds
        if startSample is None and batchAttemptCount == 0:
            headerIdleSeconds = initialHeaderIdleSeconds
        readBatchPayload(
            link,
            batchBytes,
            batchHeader,
            sizes["batchPayloadByteCount"],
            headerIdleSeconds,
            maxIdleSeconds,
        )
        try:
            batchRows, startSample = decodeBatchRows(
                batchBytes,
                sizes["batchPacketCount"],
                packetSize,
                packetFormat,
                startSample,
            )
        except ValueError as error:
            liveDiagnostics = markBadBatch(
                liveDiagnostics,
                str(error),
            )
            batchAttemptCount += 1
            if (
                liveDiagnostics["consecutiveBadBatchCount"]
                > maxConsecutiveBadBatches
            ):
                raise ValueError(
                    f"{error}. "
                    f"goodBatchCount={liveDiagnostics['goodBatchCount']} "
                    f"badBatchCount={liveDiagnostics['badBatchCount']} "
                    "live stream stayed corrupt after repeated resync "
                    "attempts."
                )
            continue
        rows.extend(batchRows)
        liveDiagnostics = markGoodBatch(liveDiagnostics)
        batchAttemptCount += 1
        batchIndex += 1
    return rows, startSample


# Builds the whole-batch count for one selected sample window.
def batchCount(sampleCount, batchPacketCount):
    if sampleCount % batchPacketCount != 0:
        raise ValueError("buffer and step sizes must align to full batches")
    return sampleCount // batchPacketCount


# Reads one fixed-size buffer window from a live binary serial stream.
def liveBuf(
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
):
    sizes = captureSizes(
        sampleRate,
        batchSeconds,
        packetSize,
        batchHeader,
    )
    sampleCount = int(sampleRate * bufferSeconds)
    bufferBatchCount = batchCount(sampleCount, sizes["batchPacketCount"])
    liveDiagnostics = liveDiagState()
    with openSerialLink(port, baudrate, timeout) as link:
        link.reset_input_buffer()
        rows, _ = readBinaryBatches(
            link,
            bufferBatchCount,
            sizes,
            batchHeader,
            packetFormat,
            packetSize,
            maxIdleSeconds,
            initialHeaderIdleSeconds,
            None,
            liveDiagnostics,
            maxConsecutiveBadBatches,
        )
    dataFrame = rowsDf(rows, sensorColumns)
    bufferData = buildBuf(
        dataFrame,
        sensorColumns,
        sampleRate,
        bufferSeconds,
        0,
    )
    return withLiveDiagnostics(bufferData, liveDiagnostics)


# Yields one normalized live batch at a time with stream diagnostics.
def liveBatchRows(
    port,
    sampleRate,
    baudrate,
    timeout,
    batchSeconds,
    batchHeader,
    packetFormat,
    packetSize,
    maxIdleSeconds,
    initialHeaderIdleSeconds,
    maxConsecutiveBadBatches,
):
    sizes = captureSizes(
        sampleRate,
        batchSeconds,
        packetSize,
        batchHeader,
    )
    liveDiagnostics = liveDiagState()
    with openSerialLink(port, baudrate, timeout) as link:
        link.reset_input_buffer()
        startSample = None
        while True:
            batchRows, startSample = readBinaryBatches(
                link,
                1,
                sizes,
                batchHeader,
                packetFormat,
                packetSize,
                maxIdleSeconds,
                initialHeaderIdleSeconds,
                startSample,
                liveDiagnostics,
                maxConsecutiveBadBatches,
            )
            yield batchRows, dict(liveDiagnostics)


# Yields rolling live buffers with a fixed-size window and stepped overlap.
def liveRoll(
    port,
    sensorColumns,
    sampleRate,
    bufferSeconds,
    stepSeconds,
    baudrate,
    timeout,
    batchSeconds,
    batchHeader,
    packetFormat,
    packetSize,
    maxIdleSeconds,
    initialHeaderIdleSeconds,
    maxConsecutiveBadBatches,
):
    sizes = captureSizes(
        sampleRate,
        batchSeconds,
        packetSize,
        batchHeader,
    )
    sampleCount = int(sampleRate * bufferSeconds)
    stepCount = int(sampleRate * stepSeconds)
    bufferBatchCount = batchCount(sampleCount, sizes["batchPacketCount"])
    stepBatchCount = batchCount(stepCount, sizes["batchPacketCount"])
    liveDiagnostics = liveDiagState()

    with openSerialLink(port, baudrate, timeout) as link:
        link.reset_input_buffer()
        initialRows, startSample = readBinaryBatches(
            link,
            bufferBatchCount,
            sizes,
            batchHeader,
            packetFormat,
            packetSize,
            maxIdleSeconds,
            initialHeaderIdleSeconds,
            None,
            liveDiagnostics,
            maxConsecutiveBadBatches,
        )
        initialFrame = rowsDf(initialRows, sensorColumns)
        bufferData = buildBuf(
            initialFrame,
            sensorColumns,
            sampleRate,
            bufferSeconds,
            0,
        )
        bufferData = withLiveDiagnostics(bufferData, liveDiagnostics)
        yield bufferData

        while True:
            newRows, startSample = readBinaryBatches(
                link,
                stepBatchCount,
                sizes,
                batchHeader,
                packetFormat,
                packetSize,
                maxIdleSeconds,
                initialHeaderIdleSeconds,
                startSample,
                liveDiagnostics,
                maxConsecutiveBadBatches,
            )
            newRowsFrame = rowsDf(newRows, sensorColumns)
            bufferData = rollBuf(
                bufferData,
                newRowsFrame,
                sensorColumns,
                sampleRate,
                bufferSeconds,
            )
            bufferData = withLiveDiagnostics(bufferData, liveDiagnostics)
            yield bufferData
