import struct
import time

import serial


sampleRate = 1000
batchSeconds = 1
captureSeconds = 60
serialPort = "/dev/cu.usbmodem1401"
baudRate = 2000000
serialTimeout = 0.2
maxIdleSeconds = 3.0
batchHeader = b"\xAA\xBB\xCC\xDD"
packetFormat = "<I14h"
packetSize = struct.calcsize(packetFormat)


# Returns the current target motor sensor column order.
def motorCols():
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
    ]
    return sensorColumns


# Builds the serial link with the minimal transport settings.
def openSerialLink(serialPort, baudRate, timeout):
    link = serial.Serial(
        port=serialPort,
        baudrate=baudRate,
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


# Builds the target packet count from the selected capture seconds.
def targetPacketCount(sampleRate, captureSeconds):
    return int(sampleRate * captureSeconds)


# Builds the current terminal status line for the live stream.
def streamStatus(rowsReceived, rowsPerSecond, lastRowValues):
    fieldCount = len(lastRowValues)
    sampleValue = lastRowValues[0]
    statusText = (
        f"rows={rowsReceived} "
        f"rowsPerSecond={rowsPerSecond:.1f} "
        f"fieldCount={fieldCount} "
        f"sample={sampleValue}"
    )
    return statusText


# Reads one fixed framed batch payload after syncing to the batch header.
def readBatchPayload(
    link,
    batchBytes,
    batchHeader,
    batchPayloadByteCount,
    maxIdleSeconds,
):
    syncToHeader(link, batchHeader, maxIdleSeconds)
    readBlock(
        link,
        batchBytes,
        batchPayloadByteCount,
        maxIdleSeconds,
    )
    return batchBytes


# Opens the serial link and prints one light live receive summary.
def runCapture(
    serialPort,
    baudRate,
    timeout,
    sampleRate,
    batchSeconds,
    captureSeconds,
    batchHeader,
    packetFormat,
    packetSize,
):
    sensorColumns = motorCols()
    sizes = captureSizes(
        sampleRate,
        batchSeconds,
        packetSize,
        batchHeader,
    )
    batchBytes = bytearray(sizes["batchPayloadByteCount"])
    targetRows = targetPacketCount(sampleRate, captureSeconds)

    print("serialPort:", serialPort)
    print("baudRate:", baudRate)
    print("sampleRateTarget:", sampleRate)
    print("packetSize:", packetSize)
    print("batchHeader:", batchHeader.hex())
    print("batchSeconds:", batchSeconds)
    print("captureSeconds:", captureSeconds)
    print("targetRows:", targetRows)
    print("sensorColumns:", sensorColumns)

    with openSerialLink(serialPort, baudRate, timeout) as link:
        link.reset_input_buffer()
        print("streaming")

        rowsReceived = 0
        startSample = None
        while True:
            reportTime = time.perf_counter()
            readBatchPayload(
                link,
                batchBytes,
                batchHeader,
                sizes["batchPayloadByteCount"],
                maxIdleSeconds,
            )
            rowsReceived += sizes["batchPacketCount"]
            firstRowValues = unpackPacketAt(batchBytes, 0, packetFormat)
            lastRowValues = unpackPacketAt(
                batchBytes,
                sizes["batchPayloadByteCount"] - packetSize,
                packetFormat,
            )
            if startSample is None:
                startSample = firstRowValues[0]
                firstRowValues = normalizeRow(firstRowValues, startSample)
                print("firstRow:", firstRowValues)
            lastRowValues = normalizeRow(lastRowValues, startSample)
            if rowsReceived == sizes["batchPacketCount"]:
                print("lastRow:", lastRowValues)
            now = time.perf_counter()
            rowsPerSecond = sizes["batchPacketCount"] / (
                now - reportTime
            )
            print(
                streamStatus(
                    rowsReceived,
                    rowsPerSecond,
                    lastRowValues,
                )
            )
            if rowsReceived >= targetRows:
                print("captureComplete")
                print("finalRow:", lastRowValues)
                return


# Runs one direct live serial capture pass.
def main():
    runCapture(
        serialPort,
        baudRate,
        serialTimeout,
        sampleRate,
        batchSeconds,
        captureSeconds,
        batchHeader,
        packetFormat,
        packetSize,
    )


if __name__ == "__main__":
    main()
