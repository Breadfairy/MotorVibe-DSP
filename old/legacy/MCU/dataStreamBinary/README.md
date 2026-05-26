# Data Stream Binary

Current ESP32-H2 quick capture firmware for the motor sensing stream.

## Current Outcome

- Target operating point: `800 Hz`
- Current validated setup reaches stable `~800 Hz`
- `2 x MPU6050` on one shared I2C bus
- Binary USB serial stream to Python host

## Current Serial Contract

- baud: `2000000`
- batch header: `AABBCCDD`
- packet format: `<I13h`
- samples per batch: `800`
- packet size: `30` bytes

Field order:

- `sample`
- `mpu1AccX`, `mpu1AccY`, `mpu1AccZ`
- `mpu1GyrX`, `mpu1GyrY`, `mpu1GyrZ`
- `mpu2AccX`, `mpu2AccY`, `mpu2AccZ`
- `mpu2GyrX`, `mpu2GyrY`, `mpu2GyrZ`
- `ds18b20`

## Current Firmware Notes

- I2C clock is set to `800000`
- sample interval is `1250 us`
- binary timing debug is currently off
- DS18B20 is not physically attached yet
- firmware currently sends `ds18b20 = 0`

## Hardware Notes

- both MPU6050 devices are currently read on the same I2C bus
- separate buses were attempted earlier but were not stable in prior testing
- this current shared-bus setup is the one that reached the validated rate

## Files

- sketch: `dataStreamBinary.ino`
- make helper: `../Makefile`

## Flash And Check

From `MCU/`:

```bash
make flash
```

For the Python-side rate check:

```bash
cd ..
python3 src/quickCapture/serialCapture.py
```

For the main live monitor:

```bash
cd ..
python3 src/main.py live
```

Current note:
- the main live plot refreshes once per incoming `1s` batch

## Known Temporary Items

- DS18B20 sensor value is a placeholder until the physical sensor is added
- if DS18B20 is added later, keep the same field position at the end of the
  packet unless the Python side is updated at the same time
