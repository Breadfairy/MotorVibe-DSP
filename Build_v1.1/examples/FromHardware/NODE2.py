import serial
import struct
import csv
import math
from datetime import datetime
import time

# ----------------- CONFIG -----------------
PORT = "COM5"
BAUD = 1000000
DURATION = 20               # seconds to record
BUFFER_SAMPLES = 32         # samples per read
SAMPLE_SIZE = 32            # bytes per sample (4 + 12*2 + 4 = 32)
SAMPLE_FORMAT = '<I12hf'    # uint32 t_us + 12x int16 IMU fields + float tempC
READY_PREFIX = "Sample struct size (bytes):"
START_COMMAND = b"START\n"
OUTPUT_CSV = f"motordata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# IMU conversion factors (MPU6050)
ACCEL_SCALE = 16384.0       # LSB/g for ±2g
GYRO_SCALE = 131.0          # LSB/(°/s) for ±250°/s

headers = [
    "t_us", "t_s",
    "ax1","ay1","az1","gx1","gy1","gz1",
    "ax2","ay2","az2","gx2","gy2","gz2",
    "tempC"
]

def convert_accel(raw):
    return raw / ACCEL_SCALE

def convert_gyro(raw):
    return raw / GYRO_SCALE

def main():
    print(f"Recording from {PORT} at {BAUD} baud for {DURATION} seconds...")
    print(f"Saving to {OUTPUT_CSV}")

    sample_count = 0
    first_t_us = None
    last_t_us = 0
    leftover = b""

    with serial.Serial(PORT, BAUD, timeout=1) as ser, open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        print("Waiting for Arduino ready signal...")
        while True:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if len(line) == 0:
                continue
            print(f"  Arduino: {line}")
            if line.startswith(READY_PREFIX):
                break

        ser.write(START_COMMAND)
        ser.flush()
        print("Stream synchronised. Recording...")

        start_time = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= DURATION:
                break

            new_bytes = ser.read(BUFFER_SAMPLES * SAMPLE_SIZE)
            if len(new_bytes) == 0:
                continue

            leftover += new_bytes

            while len(leftover) >= SAMPLE_SIZE:
                sample_bytes = leftover[:SAMPLE_SIZE]
                leftover = leftover[SAMPLE_SIZE:]
                unpacked = struct.unpack(SAMPLE_FORMAT, sample_bytes)
                t_us_raw = unpacked[0]
                if first_t_us is None:
                    first_t_us = t_us_raw
                t_us = (t_us_raw - first_t_us) & 0xFFFFFFFF
                t_s = t_us / 1e6
                last_t_us = t_us

                ax1, ay1, az1 = unpacked[1:4]
                gx1, gy1, gz1 = unpacked[4:7]
                ax2, ay2, az2 = unpacked[7:10]
                gx2, gy2, gz2 = unpacked[10:13]
                tempC = unpacked[13]
                if not math.isfinite(tempC):
                    tempC = 0.0

                ax1_g = convert_accel(ax1)
                ay1_g = convert_accel(ay1)
                az1_g = convert_accel(az1)
                gx1_dps = convert_gyro(gx1)
                gy1_dps = convert_gyro(gy1)
                gz1_dps = convert_gyro(gz1)
                ax2_g = convert_accel(ax2)
                ay2_g = convert_accel(ay2)
                az2_g = convert_accel(az2)
                gx2_dps = convert_gyro(gx2)
                gy2_dps = convert_gyro(gy2)
                gz2_dps = convert_gyro(gz2)

                row = [
                    t_us,
                    t_s,
                    ax1_g,
                    ay1_g,
                    az1_g,
                    gx1_dps,
                    gy1_dps,
                    gz1_dps,
                    ax2_g,
                    ay2_g,
                    az2_g,
                    gx2_dps,
                    gy2_dps,
                    gz2_dps,
                    tempC,
                ]

                writer.writerow(row)
                sample_count += 1

    elapsed_time = time.perf_counter() - start_time
    avg_sample_rate = sample_count / elapsed_time
    device_sample_rate = sample_count / (last_t_us / 1e6)

    print(f"\nRecording finished.")
    print(f"Captured {sample_count} samples in {elapsed_time:.2f} seconds")
    print(f"Average sample rate: {avg_sample_rate:.2f} Hz")
    print(f"Device timestamp rate: {device_sample_rate:.2f} Hz")

if __name__ == "__main__":
    main()
