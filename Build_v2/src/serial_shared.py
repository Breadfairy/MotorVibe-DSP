################################################################################
# Imports                                                                      #
################################################################################
import math
import struct
import time

################################################################################
# variables/constants                                                          #
################################################################################
# Firmware packet layout.
recordFormat = "<I12hf"
bufferSampleCount = 32
accelScale = 16384.0
gyroScale = 131.0
readyPrefix = "Sample struct size (bytes):"
startCommand = b"START\n"
recordSize = struct.calcsize(recordFormat)
bufferSize = recordSize * bufferSampleCount

################################################################################
# main functions                                                               #
################################################################################


# Resets the serial link.
def prepareLink(link):
    link.dtr = False
    link.rts = False
    time.sleep(0.2)
    link.reset_input_buffer()
    link.dtr = True
    link.rts = True


# Waits for the MCU, then starts streaming.
def waitForReady(link, verbose=False):
    if verbose:
        print("waiting for device ready...")

    while True:
        deviceLine = link.readline().decode("ascii", errors="ignore").strip()
        if len(deviceLine) == 0:
            continue
        if verbose:
            print("device:", deviceLine)
        if deviceLine.startswith(readyPrefix):
            link.write(startCommand)
            link.flush()
            return


# Decodes complete packets.
def decodePackets(byteBuffer):
    packets = []
    while len(byteBuffer) >= recordSize:
        packetBytes = byteBuffer[:recordSize]
        byteBuffer = byteBuffer[recordSize:]
        packets.append(struct.unpack(recordFormat, packetBytes))
    return packets, byteBuffer


# Converts one packet into a CSV/live row.
def packetRow(packetValues, firstTUsRaw):
    tUsRaw = packetValues[0]
    tUs = (tUsRaw - firstTUsRaw) & 0xFFFFFFFF
    tempC = packetValues[13]
    if not math.isfinite(tempC):
        tempC = 0.0

    return [
        tUs,
        tUs / 1e6,
        packetValues[1] / accelScale,
        packetValues[2] / accelScale,
        packetValues[3] / accelScale,
        packetValues[4] / gyroScale,
        packetValues[5] / gyroScale,
        packetValues[6] / gyroScale,
        packetValues[7] / accelScale,
        packetValues[8] / accelScale,
        packetValues[9] / accelScale,
        packetValues[10] / gyroScale,
        packetValues[11] / gyroScale,
        packetValues[12] / gyroScale,
        tempC,
    ]
