# Quick Capture

Minimal live binary serial capture for checking USB-C connectivity and
whether the host can keep up with the target `1000 Hz` stream.

This quick harness currently uses:

- `2000000` baud
- binary framed batches
- batch header `AABBCCDD`
- packet format `<I14h`
- fixed `15` field order

Field order:

- `sample`
- `mpu1AccX`, `mpu1AccY`, `mpu1AccZ`
- `mpu1GyrX`, `mpu1GyrY`, `mpu1GyrZ`
- `mpu1Temp`
- `mpu2AccX`, `mpu2AccY`, `mpu2AccZ`
- `mpu2GyrX`, `mpu2GyrY`, `mpu2GyrZ`
- `mpu2Temp`

## Dependencies

Install:

```bash
python3 -m pip install pyserial
```

## Spinup

1. Connect the MCU over USB-C.
2. Start sensing on the MCU.
3. Open `quickCapture/serialCapture.py`.
4. Set `serialPort` for your machine.
5. Run the script:

```bash
python3 quickCapture/serialCapture.py
```

6. Watch the terminal output:
   `firstRow` confirms the first live batch arrived.
   `rowsPerSecond` shows the receive rate on the host.
   `fieldCount` should stay at `15`.
   `sample` should keep advancing.
7. The run stops after the configured capture duration.

## Notes

- This script does not run DSP or ML.
- It prints one status line per received batch.
- Plotting and animation scripts live under `quickCapture/video/`.
- Example port values:
  `COM3` on Windows
  `/dev/cu.usbmodem*` or `/dev/cu.SLAB_USBtoUART` on macOS
