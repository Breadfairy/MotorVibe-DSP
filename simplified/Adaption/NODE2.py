import serial
import struct
import csv
from datetime import datetime
import time

# ----------------- CONFIG -----------------
PORT = "COM5"
BAUD = 921600
DURATION = 20               # seconds to record
BUFFER_SAMPLES = 32         # samples per read
SAMPLE_SIZE = 30            # bytes per sample
SAMPLE_FORMAT = '<I12hh'    # 4 + 12*2 + 2 = 30 bytes per sample
OUTPUT_CSV = f"motordata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# IMU conversion factors (assuming MPU6050)
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

def convert_temp(raw):
    return raw / 340.0 + 36.53

def main():
    print(f"Recording from {PORT} at {BAUD} baud for {DURATION} seconds...")
    print(f"Saving to {OUTPUT_CSV}")

    sample_count = 0
    start_time = time.time()

    with serial.Serial(PORT, BAUD, timeout=1) as ser, open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        while True:
            elapsed = time.time() - start_time
            if elapsed >= DURATION:
                break

            data = ser.read(BUFFER_SAMPLES * SAMPLE_SIZE)
            if len(data) % SAMPLE_SIZE != 0:
                continue  # skip incomplete buffer

            num_samples = len(data) // SAMPLE_SIZE
            for i in range(num_samples):
                start = i * SAMPLE_SIZE
                sample_bytes = data[start:start+SAMPLE_SIZE]

                try:
                    unpacked = struct.unpack(SAMPLE_FORMAT, sample_bytes)
                except struct.error:
                    continue  # skip bad sample

                t_us = unpacked[0]
                t_s = t_us / 1e6

                # Extract raw values
                ax1, ay1, az1 = unpacked[1:4]
                gx1, gy1, gz1 = unpacked[4:7]
                ax2, ay2, az2 = unpacked[7:10]
                gx2, gy2, gz2 = unpacked[10:13]
                temp_raw = unpacked[-1]

                # Sanity check: ignore clearly invalid readings
                if not (-32768 <= ax1 <= 32767 and -32768 <= gx1 <= 32767):
                    continue
                if not (-32768 <= ax2 <= 32767 and -32768 <= gx2 <= 32767):
                    continue

                # Convert to physical units
                ax1_g, ay1_g, az1_g = convert_accel(ax1), convert_accel(ay1), convert_accel(az1)
                gx1_dps, gy1_dps, gz1_dps = convert_gyro(gx1), convert_gyro(gy1), convert_gyro(gz1)

                ax2_g, ay2_g, az2_g = convert_accel(ax2), convert_accel(ay2), convert_accel(az2)
                gx2_dps, gy2_dps, gz2_dps = convert_gyro(gx2), convert_gyro(gy2), convert_gyro(gz2)

                tempC = convert_temp(temp_raw)

                row = [t_us, t_s,
                       ax1_g, ay1_g, az1_g, gx1_dps, gy1_dps, gz1_dps,
                       ax2_g, ay2_g, az2_g, gx2_dps, gy2_dps, gz2_dps,
                       tempC]

                writer.writerow(row)
                sample_count += 1

    elapsed_time = time.time() - start_time
    avg_sample_rate = sample_count / elapsed_time

    print(f"\nRecording finished.")
    print(f"Captured {sample_count} samples in {elapsed_time:.2f} seconds")
    print(f"Average sample rate: {avg_sample_rate:.2f} Hz")

if __name__ == "__main__":
    main()
