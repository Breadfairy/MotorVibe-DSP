import serial

serialPort = "COM3"
baudRate = 115200
timeout = 1.0
encoding = "utf-8"


# Builds the serial link with the minimal transport settings.
def openSerialLink(
    serialPort,
    baudRate,
    timeout,
):
    link = serial.Serial(
        port=serialPort,
        baudrate=baudRate,
        timeout=timeout,
    )
    return link


# Reads the next newline-terminated CSV payload.
def readSerialRow(link, encoding):
    rowBytes = link.readline()
    rowText = rowBytes.decode(encoding).strip()
    return rowText


# Opens the serial link and prints incoming raw rows forever.
def runSerialPrint(
    serialPort,
    baudRate,
    timeout,
    encoding,
):
    with openSerialLink(
        serialPort,
        baudRate,
        timeout,
    ) as link:
        while True:
            rowText = readSerialRow(link, encoding)
            if len(rowText) > 0:
                print(rowText)


if __name__ == "__main__":
    runSerialPrint(
        serialPort,
        baudRate,
        timeout,
        encoding,
    )
