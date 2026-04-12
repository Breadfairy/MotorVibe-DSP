import serial
import struct
import csv

ser = serial.Serial('COM5', 921600, timeout=1)

record_fmt = '<Ihhhhhh'   # uint32 + 6 int16
record_size = struct.calcsize(record_fmt)

with open('mpu6050_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['t_us', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])

    leftover = b''

    while True:
        data = ser.read(4096)
        if not data:
            continue

        leftover += data

        while len(leftover) >= record_size:
            packet = leftover[:record_size]
            leftover = leftover[record_size:]
            row = struct.unpack(record_fmt, packet)
            writer.writerow(row)
