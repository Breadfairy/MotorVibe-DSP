import numpy as np
import pandas as pd
import serial

# Converts sensor columns into NumPy arrays used by DSP.
# Expects exact columns:
# Amostra, Acel_X, Acel_Y, Acel_Z, Giro_X, Giro_Y, Giro_Z, Temperatura.
def buildSignals(data):
    sample = data["Amostra"].to_numpy(dtype=np.float64)
    Acel_X = data["Acel_X"].to_numpy(dtype=np.float64)
    Acel_Y = data["Acel_Y"].to_numpy(dtype=np.float64)
    Acel_Z = data["Acel_Z"].to_numpy(dtype=np.float64)
    Giro_X = data["Giro_X"].to_numpy(dtype=np.float64)
    Giro_Y = data["Giro_Y"].to_numpy(dtype=np.float64)
    Giro_Z = data["Giro_Z"].to_numpy(dtype=np.float64)
    Temperatura = data["Temperatura"].to_numpy(dtype=np.float64)
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


# Reads a static CSV file into a DataFrame, then reuses buildSignals.
# CSV must include columns:
# Amostra, Acel_X, Acel_Y, Acel_Z, Giro_X, Giro_Y, Giro_Z, Temperatura.
def readCSV(csv_path):
    data = pd.read_csv(csv_path)
    return buildSignals(data)


# Reads live sensor samples from the MCU over serial and reuses buildSignals.
# MCU serial output requirements:
# 1) UART settings must match this function call
#    (baudrate and framing defaults).
# 2) One sample per line, newline-terminated (\n or \r\n).
# 3) Each line must contain exactly eight comma-separated values.
# 4) Field order must be:
#    Amostra,Acel_X,Acel_Y,Acel_Z,Giro_X,Giro_Y,Giro_Z,Temperatura
# 5) Each field must be numeric text parseable as float.
# 6) MCU should emit at least `sample_count` lines for one capture cycle.
# Example MCU line:
# 1,-0.936,0.726,9.944,0.611,1.405,0.412,17.47
def liveSense(
    port,
    sample_count,
    baudrate,
    delimiter,
    timeout,
    encoding,
):
    rows = []
    with serial.Serial(port=port, baudrate=baudrate, timeout=timeout) as link:
        for _ in range(sample_count):
            line = link.readline().decode(encoding).strip()
            rows.append(line.split(delimiter))
    data = pd.DataFrame(
        rows,
        columns=[
            "Amostra",
            "Acel_X",
            "Acel_Y",
            "Acel_Z",
            "Giro_X",
            "Giro_Y",
            "Giro_Z",
            "Temperatura",
        ],
    )
    return buildSignals(data)
