################################################################################
# Imports                                                                      #
################################################################################
from pathlib import Path
import struct
import time

import pandas as pd
import serial
from serial.tools import list_ports

################################################################################
# variables/constants                                                          #
################################################################################
rootDir = Path(__file__).resolve().parents[1]
magic = 0xAABBCCDD
magicBytes = struct.pack("<I", magic)
headerFmt = "<IBBHIIII"
headerSize = struct.calcsize(headerFmt)
streamBaud = 1000000
serialTimeout = 0.2
readyText = "Ready. Waiting for MODE and START..."
startCommand = b"START\n"
stopCommand = b"STOP\n"
readyWaitSeconds = 2.0
modeCommand = {
    "all_1k": b"MODE ALL_1K\n",
    "gyro_8k": b"MODE GYRO_8K\n",
}
modeId = {
    "all_1k": 1,
    "gyro_8k": 2,
}
modeName = {
    1: "all_1k",
    2: "gyro_8k",
}
modeData = {
    "all_1k": {
        "sampleRate": 1000.0,
        "sampleRateHz": 1000,
        "frameSamples": 128,
        "sampleFmt": "<6h",
        "csvCols": [
            "sample",
            "t_s",
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ],
        "timeCols": [
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz",
        ],
        "visualCols": [
            "gx",
            "gy",
            "gz",
        ],
    },
    "gyro_8k": {
        "sampleRate": 8000.0,
        "sampleRateHz": 8000,
        "frameSamples": 256,
        "sampleFmt": "<3h",
        "csvCols": [
            "sample",
            "t_s",
            "gx",
            "gy",
            "gz",
        ],
        "timeCols": [
            "gx",
            "gy",
            "gz",
        ],
        "visualCols": [
            "gx",
            "gy",
            "gz",
        ],
    },
}
accelScale = 16384.0
gyroScale = 131.0
maxIdleSeconds = 3.0
initialIdleSeconds = 8.0

################################################################################
# helpers                                                                      #
################################################################################


# Returns the config dictionary for one stream mode.
def config(mode):
    return modeData[mode]


# Returns the current sample rate for one mode.
def sampleRate(mode):
    return config(mode)["sampleRate"]


# Returns the frame sample count for one mode.
def frameSamples(mode):
    return config(mode)["frameSamples"]


# Returns the active CSV columns for one mode.
def csvCols(mode):
    return config(mode)["csvCols"]


# Returns the active raw time columns for one mode.
def timeCols(mode):
    return config(mode)["timeCols"]


# Returns the visual-only magnitude columns for one mode.
def visualCols(mode):
    return config(mode)["visualCols"]


# Returns the active packet size for one mode.
def packetSize(mode):
    return struct.calcsize(config(mode)["sampleFmt"])


# Returns the mode name from the current CSV columns.
def modeFromCols(cols):
    if "ax" in cols:
        return "all_1k"
    return "gyro_8k"


# Detects the ESP32 USB CDC port from the current host list.
def detectPort():
    ports = list(list_ports.comports())
    for i in ports:
        if i.vid == 0x303A and i.pid == 0x1001:
            return i.device
    for i in ports:
        if "usbmodem" in i.device:
            return i.device
    return None


# Opens one serial link for framed binary capture.
def openLink(port):
    if port is None:
        port = detectPort()
    link = serial.Serial(
        port=port,
        baudrate=streamBaud,
        timeout=serialTimeout,
    )
    time.sleep(0.2)
    link.reset_input_buffer()
    return link


# Waits for the firmware ready line before sending mode commands.
def waitForReady(link):
    t0 = time.perf_counter()
    while True:
        if time.perf_counter() - t0 >= readyWaitSeconds:
            return
        try:
            line = link.readline().decode("ascii", errors="ignore").strip()
        except serial.SerialException:
            return
        if len(line) == 0:
            continue
        print("device:", line)
        if line.startswith("Ready."):
            return


# Sends one selected stream mode to the MCU.
def sendMode(link, mode):
    link.write(modeCommand[mode])
    link.flush()
    time.sleep(0.1)


# Sends the stream stop command to the MCU.
def stopStream(link):
    link.write(stopCommand)
    link.flush()
    time.sleep(0.1)


# Sends the stream start command to the MCU.
def startStream(link):
    link.write(startCommand)
    link.flush()


# Reads one exact binary block from the current serial link.
def readExact(link, size, idleSeconds):
    block = bytearray(size)
    view = memoryview(block)
    count = 0
    t0 = time.perf_counter()
    while count < size:
        n = link.readinto(view[count:size])
        if n is None:
            continue
        if n == 0:
            if time.perf_counter() - t0 >= idleSeconds:
                raise TimeoutError("serial stream stalled")
            continue
        count += n
        t0 = time.perf_counter()
    return bytes(block)


# Reads forward until the next framed batch magic is found.
def syncToMagic(link, idleSeconds):
    buf = bytearray()
    t0 = time.perf_counter()
    while True:
        b = link.read(1)
        if len(b) == 0:
            if time.perf_counter() - t0 >= idleSeconds:
                raise TimeoutError("serial sync stalled")
            continue
        buf += b
        t0 = time.perf_counter()
        if len(buf) > len(magicBytes):
            buf = buf[-len(magicBytes):]
        if bytes(buf) == magicBytes:
            return


# Decodes one header block into the shared header dictionary.
def decodeHeader(headBytes):
    vals = struct.unpack(headerFmt, headBytes)
    head = {
        "magic": vals[0],
        "version": vals[1],
        "modeId": vals[2],
        "sampleCount": vals[3],
        "sequence": vals[4],
        "startSample": vals[5],
        "sampleRateHz": vals[6],
        "payloadBytes": vals[7],
    }
    head["mode"] = modeName[head["modeId"]]
    return head


# Reads one full framed batch from the current live stream.
def readFrame(link, mode, firstFrame):
    idleSeconds = maxIdleSeconds
    if firstFrame:
        idleSeconds = initialIdleSeconds
    while True:
        syncToMagic(link, idleSeconds)
        headBytes = magicBytes + readExact(
            link,
            headerSize - 4,
            maxIdleSeconds,
        )
        head = decodeHeader(headBytes)
        payload = readExact(link, head["payloadBytes"], maxIdleSeconds)
        if head["mode"] == mode:
            return head, payload


# Decodes one framed payload into the shared row layout.
def decodeRows(head, payload):
    mode = head["mode"]
    rate = float(head["sampleRateHz"])
    fmt = config(mode)["sampleFmt"]
    rows = []
    for i, vals in enumerate(struct.iter_unpack(fmt, payload)):
        sample = head["startSample"] + i
        row = [sample, sample / rate]
        if mode == "all_1k":
            row += [
                vals[0] / accelScale,
                vals[1] / accelScale,
                vals[2] / accelScale,
                vals[3] / gyroScale,
                vals[4] / gyroScale,
                vals[5] / gyroScale,
            ]
        if mode == "gyro_8k":
            row += [
                vals[0] / gyroScale,
                vals[1] / gyroScale,
                vals[2] / gyroScale,
            ]
        rows.append(row)
    return rows


# Reads one CSV file into a dataframe.
def readCsv(csvPath):
    return pd.read_csv(csvPath)
